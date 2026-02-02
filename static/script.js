// Wildfire Detection System - Frontend

class WildfireUI {
    constructor() {
        this.elements = {
            rgbFeed: document.getElementById('rgb-feed'),
            thermalFeed: document.getElementById('thermal-feed'),
            status: document.getElementById('status'),
            confidence: document.getElementById('confidence'),
            fireCount: document.getElementById('fire-count'),
            lastUpdate: document.getElementById('last-update'),
            thermalBar: document.getElementById('thermal-bar'),
            fireBar: document.getElementById('fire-bar'),
            smokeBar: document.getElementById('smoke-bar'),
            totalBar: document.getElementById('total-bar'),
            thermalValue: document.getElementById('thermal-value'),
            fireValue: document.getElementById('fire-value'),
            smokeValue: document.getElementById('smoke-value'),
            totalValue: document.getElementById('total-value'),
            thermalPeak: document.getElementById('thermal-peak'),
            firePeak: document.getElementById('fire-peak'),
            smokePeak: document.getElementById('smoke-peak'),
            totalPeak: document.getElementById('total-peak'),
            thermalThreshold: document.getElementById('thermal-threshold'),
            fireThreshold: document.getElementById('fire-threshold'),
            smokeThreshold: document.getElementById('smoke-threshold'),
            totalThreshold: document.getElementById('total-threshold'),
            tempMin: document.getElementById('temp-min'),
            tempAvg: document.getElementById('temp-avg'),
            tempMax: document.getElementById('temp-max'),
            gpsCoords: document.getElementById('gps-coords'),
        };
        
        // Peak confidence tracking for each bar
        this.peaks = {
            thermal: 0.0,
            fire: 0.0,
            smoke: 0.0,
            total: 0.0
        };
        
        this.thresholds = {
            fire_detection: 0.4,
            thermal_weight: 0.5,
            rgb_fire_weight: 0.4,
            rgb_smoke_weight: 0.1
        };
        
        this.lastFireState = false;
        this.notificationPermission = false;
        this.updateInterval = 100; // Update every 100ms (10 fps display)
        
        this.latest_lat = 0.0;
        this.latest_lon = 0.0;

        this.init();
    }
    
    async init() {
        // Request notification permission
        await this.requestNotificationPermission();
        
        // Start update loop
        this.startUpdateLoop();
        
        console.log('Wildfire Detection UI initialized');
    }
    
    async requestNotificationPermission() {
        // Check if notifications are supported
        if (!('Notification' in window)) {
            console.warn('Notifications not supported');
            return;
        }
        
        // Request permission if not already granted
        if (Notification.permission === 'default') {
            const permission = await Notification.requestPermission();
            this.notificationPermission = permission === 'granted';
        } else {
            this.notificationPermission = Notification.permission === 'granted';
        }
        
        if (this.notificationPermission) {
            console.log('Notification permission granted');
        }
    }
    
    startUpdateLoop() {
        // Fetch updates at regular interval
        setInterval(() => this.fetchUpdate(), this.updateInterval);
        
        // Fetch immediately
        this.fetchUpdate();
    }
    
    async fetchUpdate() {
        try {
            const response = await fetch('/api/images');
            
            if (!response.ok) {
                throw new Error('Failed to fetch data');
            }
            
            const data = await response.json();
            this.updateUI(data);
            
        } catch (error) {
            console.error('Update error:', error);
            this.handleConnectionError();
        }
    }
    
    updateUI(data) {
        // Update images
        this.updateImage(this.elements.rgbFeed, data.rgb);
        this.updateImage(this.elements.thermalFeed, data.thermal);
        this.updateGPS(data.gps_data);
        
        // Update thresholds from API response
        if (data.thresholds) {
            this.thresholds = data.thresholds;
        }
        
        // Update status
        if (data.fire_detected) {
            this.elements.status.textContent = 'FIRE DETECTED';
            this.elements.status.classList.add('fire');
            
            this.showFireNotification();

        } else {
            this.elements.status.textContent = 'Monitoring';
            this.elements.status.classList.remove('fire');
        }
        
        this.lastFireState = data.fire_detected;
        
        // Update confidence
        const confidence = data.confidence || 0;
        this.elements.confidence.textContent = confidence.toFixed(2);
        
        // Update fire count
        this.elements.fireCount.textContent = data.fire_count || 0;

        // Update the gps thingy
        if (this.elements.gpsCoords !== null && this.elements.gpsCoords !== undefined) {
            this.elements.gpsCoords.textContent = `${this.latest_lat}, ${this.latest_lon}`;
        }
        
        // Update breakdown
        if (data.breakdown) {
            this.updateConfidenceBar(
                this.elements.thermalBar,
                this.elements.thermalValue,
                this.elements.thermalPeak,
                this.elements.thermalThreshold,
                data.breakdown.thermal || 0,
                'thermal',
                this.thresholds.thermal_weight
            );
            this.updateConfidenceBar(
                this.elements.fireBar,
                this.elements.fireValue,
                this.elements.firePeak,
                this.elements.fireThreshold,
                data.breakdown.rgb_fire || 0,
                'fire',
                this.thresholds.rgb_fire_weight
            );
            this.updateConfidenceBar(
                this.elements.smokeBar,
                this.elements.smokeValue,
                this.elements.smokePeak,
                this.elements.smokeThreshold,
                data.breakdown.rgb_smoke || 0,
                'smoke',
                this.thresholds.rgb_smoke_weight
            );
            this.updateConfidenceBar(
                this.elements.totalBar,
                this.elements.totalValue,
                this.elements.totalPeak,
                this.elements.totalThreshold,
                data.breakdown.total || 0,
                'total',
                this.thresholds.fire_detection
            );
        }
        
        // Update timestamp
        const now = new Date();
        this.elements.lastUpdate.textContent = now.toLocaleTimeString();
        
        // Update temperature statistics
        if (data.temperature) {
            this.elements.tempMin.textContent = data.temperature.min.toFixed(1) + '°C';
            this.elements.tempAvg.textContent = data.temperature.avg.toFixed(1) + '°C';
            this.elements.tempMax.textContent = data.temperature.max.toFixed(1) + '°C';
        } else {
            this.elements.tempMin.textContent = '--°C';
            this.elements.tempAvg.textContent = '--°C';
            this.elements.tempMax.textContent = '--°C';
        }
    }
    
    updateGPS(gps_data) {
        if (gps_data) {
            this.latest_lat = gps_data.lat;
            this.latest_lon = gps_data.lon;
        }
    }

    updateImage(imgElement, base64Data) {
        if (base64Data) {
            imgElement.src = 'data:image/jpeg;base64,' + base64Data;
            imgElement.classList.add('active');
            
            // Hide "no signal" overlay
            const noSignal = imgElement.nextElementSibling;
            if (noSignal && noSignal.classList.contains('no-signal')) {
                noSignal.style.display = 'none';
            }
        } else {
            imgElement.classList.remove('active');
            
            // Show "no signal" overlay
            const noSignal = imgElement.nextElementSibling;
            if (noSignal && noSignal.classList.contains('no-signal')) {
                noSignal.style.display = 'block';
            }
        }
    }
    
    updateConfidenceBar(barElement, valueElement, peakElement, thresholdElement, value, peakKey, thresholdValue) {
        // Clamp value between 0 and 1
        const clampedValue = Math.max(0, Math.min(1, value));
        
        // Update bar width
        barElement.style.width = (clampedValue * 100) + '%';
        
        // Update text value
        valueElement.textContent = clampedValue.toFixed(2);
        
        // Update peak confidence
        if (clampedValue > this.peaks[peakKey]) {
            this.peaks[peakKey] = clampedValue;
        }
        
        // Update peak position
        const peakPercentage = (this.peaks[peakKey] * 100);
        peakElement.style.left = peakPercentage + '%';
        
        // Update threshold line position (threshold is clamped to 0-1)
        const thresholdClamped = Math.max(0, Math.min(1, thresholdValue));
        const thresholdPercentage = (thresholdClamped * 100);
        thresholdElement.style.left = thresholdPercentage + '%';
    }
    
    showFireNotification() {
        // Only show if permission granted
        if (!this.notificationPermission) {
            return;
        }
        
        // Create notification
        const notification = new Notification('🔥 Fire Detected!', {
            body: `The wildfire detection system has detected a fire at (${this.latest_lat}, ${this.latest_lon}).`,
            icon: '/static/fire-icon.png', // Optional icon
            requireInteraction: true, // Keep notification visible
            tag: 'fire-alert' // Replace previous notifications
        });
        
        // Focus window on click
        notification.onclick = () => {
            window.focus();
            notification.close();
        };
        
        console.log('Fire notification sent');
    }
    
    handleConnectionError() {
        // Show connection error state
        this.elements.status.textContent = 'Connection Error';
        this.elements.status.classList.remove('fire');
        
        // Clear images
        this.elements.rgbFeed.classList.remove('active');
        this.elements.thermalFeed.classList.remove('active');
    }
}

// Initialize UI when page loads
document.addEventListener('DOMContentLoaded', () => {
    new WildfireUI();
});