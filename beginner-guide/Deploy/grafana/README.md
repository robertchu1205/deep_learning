# Kubernetes
## **Grafana**
### Purpose
1. Customize graphs, Create dashboard by using merics from Prometheus
2. Explore and record response json, logs from gateway, and whatever we want
### Attention
* dashboard metrics from **http://{prometheus-service-name}:9090**
    1. tensorflow serving metrics (comes from \/monitoring\/prometheus\/metrics ) starts from ":"
    <br> e.g. :tensorflow:cc:saved\_model:load\_attempt_count: **New Model Released (Model Version)**
    <br> increase(:tensorflow:core:graph_runs[1m]): **Prediction Status**, would be 0 if no prediction in present
    2. **node_exporter** show metrics of **CPU/GPU Memory Usage(dcgm\_fb\_used), GPU Utilization (dcgm\_gpu\_utilization)** 
    3.  Collect metrics of **Model Health (requests.get({tfserving-link})), Image Counter, and also gateway health**, etc. in gateway

* Explore logs from **Loki**
    1. Label important infos like SN, filename, pred_class, confidence and so on
    2. **Promtail** collects those infos from /var/log/containers/*.log

