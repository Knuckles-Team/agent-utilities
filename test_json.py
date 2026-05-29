import json

d = {"id": "mem:123"}
props_str = json.dumps(d)
parsed = json.loads(props_str)
print(type(parsed))
