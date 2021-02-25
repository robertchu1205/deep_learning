# Kubernetes
## Promtail
### Introduction
**Promtail** is an agent which ships the contents of local logs to a private Lokiinstance or Grafana Cloud. It is usuallydeployed to every machine that has applications needed to be monitored.

It primarily:
1. Discovers targets
2. Attaches labels to log streams
3. Pushes them to the Loki instance.
### [Tutorial](https://www.bookstack.cn/read/loki/clients-promtail-README.md)
### Attention
1. 
```
# create configmap from file
kubectl create configmap {configmap-name} --from-file={config.yaml-directory}
```
2. server: http_listen_port -> define a port not been used
3. **Why Daemonset**
Running a logs collection daemon on every node
A **DaemonSet** ensures that all (or some) Nodes run a copy of a Pod. As nodes are added to the cluster, Pods are added to them.

### Warning
1. **docker, json, label which followed pipeline\_stages should be in this order, cuz it would scrape by the order!!!**