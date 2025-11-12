#!/bin/bash

mkdir -p ~/.streamlit

cat << EOF > ~/.streamlit/config.toml
[browser]
gatherUsageStats = true

[server]
port = 8888
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
EOF

streamlit run app.py