# This image intentionally does NOT inherit GraphOS's FastMCP image: FastMCP
# pins mcp<2, whereas this sidecar owns the official MCP SDK v2 environment.
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6

RUN useradd --system --uid 10001 --create-home graphos
WORKDIR /app
COPY mcp_v2_gateway /app/mcp_v2_gateway
RUN pip install --no-cache-dir /app/mcp_v2_gateway
USER 10001:10001
EXPOSE 8005
ENTRYPOINT ["graphos-mcp-v2-gateway"]
