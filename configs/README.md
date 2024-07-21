# Configs

`main.yaml` contains the basic and default configuration for all
the projects:
```azure
data:
  url: DATASTORE_URL

defaults:
  - data: airbnb
  - _self_
```

while in `data` folder you may find the `airbnb.yaml` which define which datasource to use