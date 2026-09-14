// Per-unit live dashboard, adapted from the original single-unit script to
// poll a specific unit's endpoint and reflect its connection state.

class UnitDashboard {
    constructor(unitId) {
        this.unitId = unitId;
        this.elements = {
            rgbFeed: document.getElementById('rgb-feed'),
            thermalFeed: document.getElementById('thermal-feed'),
            connDot: document.getElementById('conn-dot'),
            connLabel: document.getElementById('conn-label'),
            awaitingBanner: document.getElementById('awaiting-banner'),
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
            thermalThreshold: document.getElementById('thermal-threshold'),
            fireThreshold: document.getElementById('fire-threshold'),
            smokeThreshold: document.getElementById('smoke-threshold'),
            totalThreshold: document.getElementById('total-threshold'),
            tempMin: document.getElementById('temp-min'),
            tempAvg: document.getElementById('temp-avg'),
            tempMax: document.getElementById('temp-max'),
            gpsCoords: document.getElementById('gps-coords'),
        };

        this.thresholds = {
            fire_detection: 0.4,
            thermal_weight: 0.5,
            rgb_fire_weight: 0.4,
            rgb_smoke_weight: 0.1
        };

        this.notificationPermission = false;
        this.activeNotification = null;
        this.updateInterval = 100;

        this.init();
    }

    async init() {
        await this.requestNotificationPermission();
        this.startUpdateLoop();
    }

    async requestNotificationPermission() {
        if (!('Notification' in window)) return;
        if (Notification.permission === 'default') {
            const permission = await Notification.requestPermission();
            this.notificationPermission = permission === 'granted';
        } else {
            this.notificationPermission = Notification.permission === 'granted';
        }
    }

    startUpdateLoop() {
        setInterval(() => this.fetchUpdate(), this.updateInterval);
        this.fetchUpdate();
    }

    async fetchUpdate() {
        try {
            const response = await fetch(`/api/units/${encodeURIComponent(this.unitId)}/images`);
            if (!response.ok) throw new Error('Failed to fetch data');
            const data = await response.json();
            this.updateUI(data);
        } catch (error) {
            this.handleConnectionError();
        }
    }

    updateUI(data) {
        this.updateImage(this.elements.rgbFeed, data.rgb);
        this.updateImage(this.elements.thermalFeed, data.thermal);
        this.updateGPS(data.gps_data);

        if (data.thresholds) this.thresholds = data.thresholds;

        this.elements.awaitingBanner.hidden = !!data.connected || data.status === 'offline';

        if (!data.connected) {
            this.elements.connDot.className = 'status-dot ' + (data.status === 'offline' ? 'offline' : '');
            this.elements.connLabel.textContent = data.status === 'offline' ? 'Offline' : 'Awaiting connection';
        } else if (data.fire_detected) {
            this.elements.connDot.className = 'status-dot fire pulse';
            this.elements.connLabel.textContent = 'FIRE DETECTED';
            if (!this.activeNotification) this.showFireNotification();
        } else {
            this.elements.connDot.className = 'status-dot online';
            this.elements.connLabel.textContent = 'Monitoring';
        }

        const confidence = data.confidence || 0;
        this.elements.confidence.textContent = confidence.toFixed(2);
        this.elements.fireCount.textContent = data.fire_count || 0;

        if (this.elements.gpsCoords && data.gps_data) {
            this.elements.gpsCoords.textContent = `${data.gps_data.lat}, ${data.gps_data.lon}`;
        } else if (this.elements.gpsCoords) {
            this.elements.gpsCoords.textContent = '(-, -)';
        }

        if (data.breakdown) {
            this.updateConfidenceBar(this.elements.thermalBar, this.elements.thermalValue, this.elements.thermalThreshold, data.breakdown.thermal || 0, this.thresholds.thermal_weight);
            this.updateConfidenceBar(this.elements.fireBar, this.elements.fireValue, this.elements.fireThreshold, data.breakdown.rgb_fire || 0, this.thresholds.rgb_fire_weight);
            this.updateConfidenceBar(this.elements.smokeBar, this.elements.smokeValue, this.elements.smokeThreshold, data.breakdown.rgb_smoke || 0, this.thresholds.rgb_smoke_weight);
            this.updateConfidenceBar(this.elements.totalBar, this.elements.totalValue, this.elements.totalThreshold, data.breakdown.total || 0, this.thresholds.fire_detection);
        }

        const now = new Date();
        this.elements.lastUpdate.textContent = now.toLocaleTimeString();

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

    updateGPS(gpsData) {
        this.latestGps = gpsData || null;
    }

    updateImage(imgElement, base64Data) {
        if (base64Data) {
            imgElement.src = 'data:image/jpeg;base64,' + base64Data;
            imgElement.classList.add('active');
            const noSignal = imgElement.nextElementSibling;
            if (noSignal && noSignal.classList.contains('no-signal')) noSignal.style.display = 'none';
        } else {
            imgElement.classList.remove('active');
            const noSignal = imgElement.nextElementSibling;
            if (noSignal && noSignal.classList.contains('no-signal')) noSignal.style.display = 'block';
        }
    }

    updateConfidenceBar(barElement, valueElement, thresholdElement, value, thresholdValue) {
        const clampedValue = Math.max(0, Math.min(1, value));
        barElement.style.width = (clampedValue * 100) + '%';
        valueElement.textContent = clampedValue.toFixed(2);
        const thresholdClamped = Math.max(0, Math.min(1, thresholdValue));
        thresholdElement.style.left = (thresholdClamped * 100) + '%';
    }

    showFireNotification() {
        if (!this.notificationPermission) return;
        const lat = this.latestGps ? this.latestGps.lat : '-';
        const lon = this.latestGps ? this.latestGps.lon : '-';
        const notification = new Notification('Fire Detected', {
            body: `Unit ${this.unitId} detected a fire at (${lat}, ${lon}).`,
            requireInteraction: true
        });
        this.activeNotification = notification;
        notification.onclose = () => { this.activeNotification = null; };
        notification.onclick = () => { window.focus(); notification.close(); };
    }

    handleConnectionError() {
        this.elements.connLabel.textContent = 'Connection error';
        this.elements.connDot.className = 'status-dot offline';
        this.elements.rgbFeed.classList.remove('active');
        this.elements.thermalFeed.classList.remove('active');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const unitId = document.body.dataset.unitId;
    new UnitDashboard(unitId);
});
