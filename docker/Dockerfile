FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . /app/
RUN pip install --no-cache-dir -e .[all]

EXPOSE 8004

CMD ["python", "-m", "agent_utilities.mcp.kg_server"]
