# 🎤 Oreja Speaker Analytics Dashboard

A comprehensive GUI application for exploring and visualizing the Oreja speaker database and learning progress.

## 📊 Features

### **Overview Dashboard**
- **Real-time Speaker List**: Browse all speakers with type, sample count, confidence, and last activity
- **Live Statistics**: Database stats, quality metrics, recent activity, and storage information
- **Color-coded Types**: Visual distinction between auto-generated, user-corrected, and enrolled speakers
- **Auto-refresh**: Continuous updates as the system learns

### **Speaker Details View** 
- **Individual Profiles**: Deep dive into any speaker's learning journey
- **Learning Progress Charts**:
  - Confidence evolution over time
  - Confidence distribution histogram
  - Audio length patterns
  - Cumulative learning progress
- **Quality Assessment**: Recognition quality, confidence trends, and training completeness
- **Timeline Analysis**: First/last samples, learning span, samples per day

### **Learning Analytics**
- **System-wide Visualizations**:
  - Speaker type distribution (pie chart)
  - Confidence distribution across all speakers
  - Training data distribution
  - Top 10 most trained speakers
  - Confidence vs training data correlation
  - Recent activity timeline
- **Exportable Data**: CSV export for external analysis
- **Time Range Filtering**: Focus on specific periods

### **Database Management**
- **Cleanup Tools**: Remove low-quality or empty speakers
- **Export Functions**: JSON database export, CSV analytics export
- **Report Generation**: Comprehensive analytics reports
- **Database Reset**: Complete data reset (with safety confirmations)
- **Management Logging**: Track all administrative actions

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8+ with tkinter support
- Oreja backend server running (`python backend/server.py`)
- Speaker database with some data

### **Installation & Launch**

#### **Option 1: Easy Launcher (Recommended)**
```bash
cd backend
python start_analytics.py
```

The launcher will:
- Check for required dependencies
- Install missing packages automatically
- Verify tkinter availability
- Launch the dashboard

#### **Option 2: Manual Installation**
```bash
# Install dependencies
pip install matplotlib numpy pandas requests

# Launch dashboard
cd backend
python speaker_analytics_gui.py
```

#### **Option 3: Using Requirements**
```bash
# Install all dependencies
pip install -r backend/requirements.txt

# Launch dashboard
cd backend
python speaker_analytics_gui.py
```

## 📈 Understanding the Visualizations

### **Speaker Learning Progress**

**Confidence Over Time Chart**
- Shows how speaker recognition improves with more samples
- Upward trend indicates successful learning
- Plateaus suggest the speaker is well-trained

**Confidence Distribution**
- Histogram showing the spread of confidence scores
- Narrow distribution around high values = consistent recognition
- Wide distribution = inconsistent or learning speaker

**Learning Progress (Cumulative)**
- Linear growth = steady training
- Steep increases = intensive training sessions
- Flat periods = no recent activity

### **System Analytics**

**Speaker Type Distribution**
- 🤖 Auto: System-generated speakers (need correction)
- ✅ Corrected: User-corrected speakers (learning targets)
- 👤 Enrolled: Manually enrolled speakers
- Helps assess correction workflow effectiveness

**Confidence vs Training Data Scatter**
- X-axis: Number of audio samples
- Y-axis: Average confidence
- Ideal pattern: upward trend (more data = higher confidence)
- Outliers may indicate speakers needing attention

## 🎯 Use Cases

### **For Developers**
- **Monitor Learning Performance**: Track how well the speaker learning system works
- **Debug Recognition Issues**: Identify speakers with poor confidence trends
- **Optimize Training**: Find speakers needing more samples or cleanup
- **System Health**: Overall database statistics and trends

### **For Users**
- **Understand Recognition Quality**: See how well the system knows each speaker
- **Track Improvement**: Visualize learning progress over time
- **Manage Speaker Database**: Clean up, merge, or remove speakers
- **Export Data**: Generate reports for analysis or sharing

### **For Research**
- **Learning Curve Analysis**: Study how speaker recognition improves
- **Data Quality Assessment**: Identify patterns in training data
- **System Evaluation**: Measure overall learning system performance
- **Temporal Analysis**: Understand learning patterns over time

## 🔧 Advanced Features

### **Real-time Monitoring**
- **Auto-refresh**: Enable continuous updates (5-second intervals)
- **Live Statistics**: See changes as new audio is processed
- **Activity Timeline**: Track recent speaker interactions

### **Data Export & Reporting**
- **JSON Export**: Complete database dump for backup/analysis
- **CSV Analytics**: Speaker statistics for spreadsheet analysis
- **Comprehensive Reports**: Detailed text reports with all metrics
- **Chart Export**: Save visualizations as images

### **Database Management**
- **Smart Cleanup**: Remove speakers with <3 samples automatically
- **Quality Filtering**: Focus on high/medium/low confidence speakers
- **Bulk Operations**: Manage multiple speakers efficiently
- **Safe Reset**: Complete database reset with multiple confirmations

## 📊 Key Metrics Explained

### **Speaker Quality Indicators**
- **Recognition Quality**: 
  - ⭐⭐⭐ Excellent (>0.8 confidence)
  - ⭐⭐ Good (0.6-0.8 confidence)
  - ⭐ Fair (0.4-0.6 confidence)
  - ❌ Poor (<0.4 confidence)

- **Training Completeness**:
  - ✅ Well-trained (>50 samples)
  - ⚠️ Moderately trained (20-50 samples)
  - ⚠️ Lightly trained (5-20 samples)
  - ❌ Insufficient (<5 samples)

- **Confidence Trends**:
  - ↗️ Improving (recent > early confidence)
  - ↘️ Declining (recent < early confidence)
  - ➡️ Stable (consistent confidence)

### **System Health Metrics**
- **Active Speakers**: Speakers used in last 24h/7d
- **Sample Distribution**: How training data is spread
- **Confidence Distribution**: Overall recognition quality
- **Database Growth**: Learning progress over time

## 🚨 Troubleshooting

### **Common Issues**

**"tkinter not found"**
- **Windows/Mac**: Reinstall Python with tkinter support
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **CentOS/RHEL**: `sudo yum install tkinter`

**"Backend connection failed"**
- Ensure backend server is running: `python backend/server.py`
- Check backend URL in dashboard (default: http://127.0.0.1:8000)
- Verify firewall/network settings

**"No speaker data available"**
- Run some transcriptions first to generate speaker data
- Check that `speaker_data/speaker_database.db` exists
- Verify database permissions

**Charts not displaying**
- Install matplotlib: `pip install matplotlib`
- Check for GUI backend issues: try different matplotlib backends
- Verify display environment variables (Linux)

### **Performance Tips**
- **Large Databases**: Disable auto-refresh for better performance
- **Slow Loading**: Close other applications using matplotlib
- **Memory Usage**: Restart dashboard periodically for large datasets

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Audio Visualization**: Live waveform displays
- **Speaker Similarity Analysis**: Find similar speakers
- **Training Recommendations**: Suggest speakers needing attention
- **Custom Analytics**: User-defined metrics and charts
- **Web Interface**: Browser-based dashboard
- **Integration with Main App**: Embedded analytics in Oreja UI

### **Advanced Analytics**
- **Machine Learning Insights**: Cluster analysis, anomaly detection
- **Temporal Patterns**: Daily/weekly activity patterns
- **Quality Prediction**: Predict future confidence based on trends
- **Comparative Analysis**: Compare speakers across different time periods

## 📝 Contributing

Contributions to the Speaker Analytics Dashboard are welcome! Areas for improvement:

- **UI/UX Enhancements**: Better layouts, themes, responsive design
- **New Visualizations**: Additional chart types and metrics
- **Performance Optimizations**: Faster loading, better memory usage
- **Export Formats**: Additional export options (PDF, Excel, etc.)
- **Documentation**: Examples, tutorials, best practices

## 📄 License

This Speaker Analytics Dashboard is part of the Oreja project and follows the same licensing terms.

---

**Happy analyzing! 🎤📊** 