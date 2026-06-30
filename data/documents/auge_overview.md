# AUGE Intelligence Platform

AUGE Intelligence provides real-time anomaly detection for industrial sensor networks. The platform ingests telemetry from temperature, pressure, and vibration sensors deployed across manufacturing facilities.

## Core Capabilities

- **Anomaly Detection**: Statistical and ML-based models flag deviations from normal operating ranges.
- **Alerting**: Configurable thresholds trigger notifications via email, Slack, or webhook.
- **Dashboards**: Operators visualize sensor trends and incident history.

## Architecture

Data flows from edge gateways to the AUGE cloud (or on-premise deployment), where stream processors compute rolling statistics and run detection models. Results are stored in a time-series database and surfaced through the web UI.

## Supported Sensors

See `sensor_catalog.md` for the full list of supported sensor types and protocols.
