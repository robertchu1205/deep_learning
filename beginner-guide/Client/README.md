# HTTP & gRPC Performance in remote or local server
<img src="./performance_imgps.png" width="700">
<img src="./performance_reqps.png" width="700">

# [Request](./request_format.json) & [Response format Json](./response_format.json)
#### Purpose
Template providing update information of program in AOI devices for AOI developers 

# The pros and cons of HTTP & gRPC 
<img src="./grpchttp_p&c.png" width="500">

# [HTTP (Through Json)](./2m_rest.py)
#### Purpose
1. **Test tensorflow serving:**
    <br> Simulate client to send request to tensorflow serving for making sure our serving is working
2. **Test model accuracy:**
    <br> Whether prediction is the same as testing the trained model by source code or not

### Detail
* requests.post(TFSERVER_URL, data=data).text -> For debugging
* {host address}:{expose port for restful}/v1/models/model/metadata
    <br> -> adding /metadata/ in the end can show what inputs, output, signature name tensorflow serving needs

### Attention
###### 1. Image information passing is quicker with base64 ( String format )
###### 2. Request default: 
model_spec.name = "model"
<br> model_spec.signature = “serving-default“

# [gRPC (Through Protobuf)](./2m-grpc.py)
#### Purpose
1. **Test tensorflow serving:**
    <br> Simulate client to send request to tensorflow serving for making sure our serving is working
2. **Test model accuracy:**
    <br> Whether prediction is the same as testing the trained model by source code or not

### Detail
$ export GRPC_TRACE=round_robin,connectivity_state,pick_first
$ export GRPC_VERBOSITY=debug
-> It will show details when the client sends inf through grpc ( helpful for debugging )

### Attention
###### TFSERVER\_URL\_GRPC = {host address}:{port for grpc}
For example: 10.41.65.77:8590 <br>
It will occur “Failed to create subchannel” if adding http:// in the front

