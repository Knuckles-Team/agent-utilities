# Database environments

The authoritative graph, RDF ontology and SPARQL query path are owned by
[epistemic-graph](../architecture/owl_rdf_layer.md). Agent Utilities runs the
control plane and uses the engine through a verified graph session. Its former
`setup-databases` command and Stardog/Fuseki backend selectors were retired;
old examples that upload ontologies to Stardog or mirror writes into Fuseki no
longer describe a supported AU path.

## Configure the engine

Use the Graph OS deployment configuration to connect to the epistemic-graph
service. Keep tenant identity and authorization in the verified session. The
engine's native RDF methods provide graph-scoped load, read, query and
reasoning. A direct connection to an external SPARQL store is a foreign source
through engine federation only where that source kind has a served contract;
unsupported sink writes fail closed.

## Optional external services

PostgreSQL/AGE and other external databases are separate integration or
federation sources. Their lifecycle is not controlled by an AU database setup
command. The separately managed Apache Jena service may still appear in the
service dashboard and deployment doctor, which check its reachability. Those
checks do not mean AU publishes its ontology there.

An external ontology distribution workflow needs a real consumer and a
versioned, authorized, atomic publication contract. Do not replace a graph by
clearing it and then loading triples in two requests: a failed second request
would leave the published graph empty.
