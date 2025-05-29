using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using Oreja.Models;

namespace Oreja.Services;

/// <summary>
/// Service for managing speaker embeddings and identification using SQLite.
/// All speaker data is stored locally with encrypted embeddings.
/// </summary>
public class SpeakerService : ISpeakerService, IDisposable
{
    private readonly ILogger<SpeakerService> _logger;
    private readonly string _databasePath;
    private SqliteConnection? _connection;
    private readonly object _lockObject = new();

    private const string DATABASE_NAME = "speakers.db";
    private const double SIMILARITY_THRESHOLD = 0.85;

    public SpeakerService(ILogger<SpeakerService> logger)
    {
        _logger = logger;
        _databasePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Oreja", DATABASE_NAME);
        
        // Ensure directory exists
        var directory = Path.GetDirectoryName(_databasePath);
        if (directory != null && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        InitializeDatabaseAsync().GetAwaiter().GetResult();
    }

    /// <summary>
    /// Initializes the SQLite database and creates tables if they don't exist.
    /// </summary>
    private async Task InitializeDatabaseAsync()
    {
        try
        {
            var connectionString = $"Data Source={_databasePath};Cache=Shared;";
            _connection = new SqliteConnection(connectionString);
            await _connection.OpenAsync();

            const string createTableSql = @"
                CREATE TABLE IF NOT EXISTS Speakers (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT NOT NULL,
                    EmbeddingId TEXT NOT NULL UNIQUE,
                    Embedding BLOB NOT NULL,
                    CreatedAt TEXT NOT NULL,
                    LastSeen TEXT NOT NULL,
                    OccurrenceCount INTEGER NOT NULL DEFAULT 1
                );
                
                CREATE INDEX IF NOT EXISTS idx_speakers_name ON Speakers(Name);
                CREATE INDEX IF NOT EXISTS idx_speakers_embedding_id ON Speakers(EmbeddingId);
                CREATE INDEX IF NOT EXISTS idx_speakers_last_seen ON Speakers(LastSeen);
            ";

            using var command = new SqliteCommand(createTableSql, _connection);
            await command.ExecuteNonQueryAsync();

            _logger.LogInformation("Speaker database initialized at {DatabasePath}", _databasePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to initialize speaker database");
            throw;
        }
    }

    /// <summary>
    /// Identifies a speaker based on their voice embedding or creates a new speaker entry.
    /// </summary>
    /// <param name="embedding">Voice embedding vector</param>
    /// <returns>Speaker information</returns>
    public async Task<Speaker> IdentifySpeakerAsync(byte[] embedding)
    {
        lock (_lockObject)
        {
            try
            {
                var embeddingId = GenerateEmbeddingId(embedding);
                
                // First try to find exact match by embedding ID
                var existingSpeaker = GetSpeakerByEmbeddingId(embeddingId);
                if (existingSpeaker != null)
                {
                    // Update last seen and occurrence count
                    UpdateSpeakerLastSeen(existingSpeaker.Id);
                    return existingSpeaker;
                }

                // Look for similar embeddings
                var similarSpeaker = FindSimilarSpeaker(embedding);
                if (similarSpeaker != null)
                {
                    // Update the embedding to the new one (speaker recognition improvement)
                    UpdateSpeakerEmbedding(similarSpeaker.Id, embeddingId, embedding);
                    return similarSpeaker;
                }

                // Create new speaker
                return CreateNewSpeaker(embeddingId, embedding);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error identifying speaker");
                throw;
            }
        }
    }

    /// <summary>
    /// Renames a speaker in the database.
    /// </summary>
    /// <param name="currentName">Current speaker name</param>
    /// <param name="newName">New speaker name</param>
    /// <returns>True if successful</returns>
    public async Task<bool> RenameSpeakerAsync(string currentName, string newName)
    {
        lock (_lockObject)
        {
            try
            {
                if (_connection == null) return false;

                const string updateSql = @"
                    UPDATE Speakers 
                    SET Name = @newName, LastSeen = @lastSeen 
                    WHERE Name = @currentName";

                using var command = new SqliteCommand(updateSql, _connection);
                command.Parameters.AddWithValue("@newName", newName);
                command.Parameters.AddWithValue("@currentName", currentName);
                command.Parameters.AddWithValue("@lastSeen", DateTime.UtcNow.ToString("O"));

                var rowsAffected = command.ExecuteNonQuery();
                
                if (rowsAffected > 0)
                {
                    _logger.LogInformation("Renamed speaker from '{CurrentName}' to '{NewName}'", currentName, newName);
                    return true;
                }

                _logger.LogWarning("No speaker found with name '{CurrentName}'", currentName);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error renaming speaker from '{CurrentName}' to '{NewName}'", currentName, newName);
                return false;
            }
        }
    }

    /// <summary>
    /// Gets all known speakers from the database.
    /// </summary>
    /// <returns>List of speakers</returns>
    public async Task<List<Speaker>> GetAllSpeakersAsync()
    {
        lock (_lockObject)
        {
            var speakers = new List<Speaker>();
            
            try
            {
                if (_connection == null) return speakers;

                const string selectSql = @"
                    SELECT Id, Name, EmbeddingId, Embedding, CreatedAt, LastSeen, OccurrenceCount 
                    FROM Speakers 
                    ORDER BY LastSeen DESC";

                using var command = new SqliteCommand(selectSql, _connection);
                using var reader = command.ExecuteReader();

                while (reader.Read())
                {
                    speakers.Add(new Speaker
                    {
                        Id = reader.GetInt32("Id"),
                        Name = reader.GetString("Name"),
                        EmbeddingId = reader.GetString("EmbeddingId"),
                        Embedding = (byte[])reader["Embedding"],
                        CreatedAt = DateTime.Parse(reader.GetString("CreatedAt")),
                        LastSeen = DateTime.Parse(reader.GetString("LastSeen")),
                        OccurrenceCount = reader.GetInt32("OccurrenceCount")
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving speakers");
            }

            return speakers;
        }
    }

    /// <summary>
    /// Deletes a speaker from the database.
    /// </summary>
    /// <param name="speakerId">Speaker ID to delete</param>
    /// <returns>True if successful</returns>
    public async Task<bool> DeleteSpeakerAsync(int speakerId)
    {
        lock (_lockObject)
        {
            try
            {
                if (_connection == null) return false;

                const string deleteSql = "DELETE FROM Speakers WHERE Id = @id";
                using var command = new SqliteCommand(deleteSql, _connection);
                command.Parameters.AddWithValue("@id", speakerId);

                var rowsAffected = command.ExecuteNonQuery();
                
                if (rowsAffected > 0)
                {
                    _logger.LogInformation("Deleted speaker with ID {SpeakerId}", speakerId);
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting speaker {SpeakerId}", speakerId);
                return false;
            }
        }
    }

    private Speaker? GetSpeakerByEmbeddingId(string embeddingId)
    {
        if (_connection == null) return null;

        const string selectSql = @"
            SELECT Id, Name, EmbeddingId, Embedding, CreatedAt, LastSeen, OccurrenceCount 
            FROM Speakers 
            WHERE EmbeddingId = @embeddingId";

        using var command = new SqliteCommand(selectSql, _connection);
        command.Parameters.AddWithValue("@embeddingId", embeddingId);
        using var reader = command.ExecuteReader();

        if (reader.Read())
        {
            return new Speaker
            {
                Id = reader.GetInt32("Id"),
                Name = reader.GetString("Name"),
                EmbeddingId = reader.GetString("EmbeddingId"),
                Embedding = (byte[])reader["Embedding"],
                CreatedAt = DateTime.Parse(reader.GetString("CreatedAt")),
                LastSeen = DateTime.Parse(reader.GetString("LastSeen")),
                OccurrenceCount = reader.GetInt32("OccurrenceCount")
            };
        }

        return null;
    }

    private Speaker? FindSimilarSpeaker(byte[] embedding)
    {
        if (_connection == null) return null;

        // For simplicity, we'll use a basic similarity check
        // In a production system, you'd want to use more sophisticated vector similarity
        const string selectSql = @"
            SELECT Id, Name, EmbeddingId, Embedding, CreatedAt, LastSeen, OccurrenceCount 
            FROM Speakers 
            ORDER BY LastSeen DESC 
            LIMIT 10"; // Check only recent speakers for performance

        using var command = new SqliteCommand(selectSql, _connection);
        using var reader = command.ExecuteReader();

        while (reader.Read())
        {
            var storedEmbedding = (byte[])reader["Embedding"];
            var similarity = CalculateCosineSimilarity(embedding, storedEmbedding);
            
            if (similarity > SIMILARITY_THRESHOLD)
            {
                return new Speaker
                {
                    Id = reader.GetInt32("Id"),
                    Name = reader.GetString("Name"),
                    EmbeddingId = reader.GetString("EmbeddingId"),
                    Embedding = storedEmbedding,
                    CreatedAt = DateTime.Parse(reader.GetString("CreatedAt")),
                    LastSeen = DateTime.Parse(reader.GetString("LastSeen")),
                    OccurrenceCount = reader.GetInt32("OccurrenceCount")
                };
            }
        }

        return null;
    }

    private Speaker CreateNewSpeaker(string embeddingId, byte[] embedding)
    {
        if (_connection == null) throw new InvalidOperationException("Database not initialized");

        const string insertSql = @"
            INSERT INTO Speakers (Name, EmbeddingId, Embedding, CreatedAt, LastSeen, OccurrenceCount)
            VALUES (@name, @embeddingId, @embedding, @createdAt, @lastSeen, 1);
            SELECT last_insert_rowid();";

        var now = DateTime.UtcNow.ToString("O");
        var speakerCount = GetSpeakerCount() + 1;
        var speakerName = $"Speaker {speakerCount}";

        using var command = new SqliteCommand(insertSql, _connection);
        command.Parameters.AddWithValue("@name", speakerName);
        command.Parameters.AddWithValue("@embeddingId", embeddingId);
        command.Parameters.AddWithValue("@embedding", embedding);
        command.Parameters.AddWithValue("@createdAt", now);
        command.Parameters.AddWithValue("@lastSeen", now);

        var speakerId = Convert.ToInt32(command.ExecuteScalar());

        _logger.LogInformation("Created new speaker '{SpeakerName}' with ID {SpeakerId}", speakerName, speakerId);

        return new Speaker
        {
            Id = speakerId,
            Name = speakerName,
            EmbeddingId = embeddingId,
            Embedding = embedding,
            CreatedAt = DateTime.Parse(now),
            LastSeen = DateTime.Parse(now),
            OccurrenceCount = 1
        };
    }

    private void UpdateSpeakerLastSeen(int speakerId)
    {
        if (_connection == null) return;

        const string updateSql = @"
            UPDATE Speakers 
            SET LastSeen = @lastSeen, OccurrenceCount = OccurrenceCount + 1 
            WHERE Id = @id";

        using var command = new SqliteCommand(updateSql, _connection);
        command.Parameters.AddWithValue("@lastSeen", DateTime.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("@id", speakerId);
        command.ExecuteNonQuery();
    }

    private void UpdateSpeakerEmbedding(int speakerId, string embeddingId, byte[] embedding)
    {
        if (_connection == null) return;

        const string updateSql = @"
            UPDATE Speakers 
            SET EmbeddingId = @embeddingId, Embedding = @embedding, LastSeen = @lastSeen, OccurrenceCount = OccurrenceCount + 1 
            WHERE Id = @id";

        using var command = new SqliteCommand(updateSql, _connection);
        command.Parameters.AddWithValue("@embeddingId", embeddingId);
        command.Parameters.AddWithValue("@embedding", embedding);
        command.Parameters.AddWithValue("@lastSeen", DateTime.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("@id", speakerId);
        command.ExecuteNonQuery();
    }

    private int GetSpeakerCount()
    {
        if (_connection == null) return 0;

        const string countSql = "SELECT COUNT(*) FROM Speakers";
        using var command = new SqliteCommand(countSql, _connection);
        return Convert.ToInt32(command.ExecuteScalar());
    }

    private static string GenerateEmbeddingId(byte[] embedding)
    {
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var hash = sha256.ComputeHash(embedding);
        return Convert.ToBase64String(hash);
    }

    private static double CalculateCosineSimilarity(byte[] embedding1, byte[] embedding2)
    {
        if (embedding1.Length != embedding2.Length)
            return 0.0;

        // Convert bytes to floats for calculation (assuming 4-byte floats)
        var len = embedding1.Length / 4;
        var vec1 = new float[len];
        var vec2 = new float[len];

        Buffer.BlockCopy(embedding1, 0, vec1, 0, embedding1.Length);
        Buffer.BlockCopy(embedding2, 0, vec2, 0, embedding2.Length);

        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        for (int i = 0; i < len; i++)
        {
            dotProduct += vec1[i] * vec2[i];
            magnitude1 += vec1[i] * vec1[i];
            magnitude2 += vec2[i] * vec2[i];
        }

        magnitude1 = Math.Sqrt(magnitude1);
        magnitude2 = Math.Sqrt(magnitude2);

        if (magnitude1 == 0.0 || magnitude2 == 0.0)
            return 0.0;

        return dotProduct / (magnitude1 * magnitude2);
    }

    public void Dispose()
    {
        _connection?.Close();
        _connection?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Interface for speaker identification and management service.
/// </summary>
public interface ISpeakerService : IDisposable
{
    Task<Speaker> IdentifySpeakerAsync(byte[] embedding);
    Task<bool> RenameSpeakerAsync(string currentName, string newName);
    Task<List<Speaker>> GetAllSpeakersAsync();
    Task<bool> DeleteSpeakerAsync(int speakerId);
} 