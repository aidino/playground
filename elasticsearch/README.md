# Elastic search

## Download and install

Source: https://www.elastic.co/guide/en/elasticsearch/reference/8.11/targz.html

### Linux

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.4-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.4-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-8.11.4-linux-x86_64.tar.gz.sha512 
tar -xzf elasticsearch-8.11.4-linux-x86_64.tar.gz
cd elasticsearch-8.11.4/ 

export ES_HOME=$PWD

# Run the following command to start Elasticsearch from the command line:
./bin/elasticsearch

export ELASTIC_PASSWORD="your_password"

```

### MacOs

```bash
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.4-darwin-x86_64.tar.gz
curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.4-darwin-x86_64.tar.gz.sha512 | shasum -a 512 -c - 
tar -xzf elasticsearch-8.11.4-darwin-x86_64.tar.gz
cd elasticsearch-8.11.4/ 

export ES_HOME=$PWD

# Run the following command to start Elasticsearch from the command line:
./bin/elasticsearch

export ELASTIC_PASSWORD="your_password"
```

You can test that your Elasticsearch node is running by sending an HTTPS request to port 9200 on localhost:
```bash
curl --cacert $ES_HOME/config/certs/http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200 
```