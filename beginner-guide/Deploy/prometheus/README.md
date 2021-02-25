# Kubernetes
## **Prometheus**
### Purpose
Acquire metrics from tensorflow serving, node exporter and gateway... <br>
in order to draw graphs in grafana
### Attention
1. the directory in "volumes/hostPath" should **"chown nobody/nogroup"**, not be root/root
2. While defining image in yaml, we make sure the image is in every node. <br> 
Otherwise, the pod can not be create successfully because the image does not exist.
3. **{prometheus\_external\_IPs}:{NodePort}/targets** provides connected status of every job_name which we define in prometheus.yml
4. In prometheus.yml, **scrape_configs** define **job_name**, **metrics_path** which is the address after the link, and **static_configs - targets** which follows with the link we want to oversee