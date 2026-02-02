# Clinical-Trial-Analytics-Dashboard
Developed a full-stack data workflow that automates ClinicalTrials.gov API extraction, data transformation, and visualization via analytical dashboards.




docker compose up -d


# How to run it
In a bash terminal go to root path of the repository where Docker-compose.yml is allocated and proceed

1. Build the images
```bash
docker compose build
```
2.Start the services
```bash
docker compose up -d
```
3. Download the model so the API can use it
```bash
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2


