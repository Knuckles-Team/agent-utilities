import sys

from rdflib import Graph, Namespace


def format_ttl(file_path):
    g = Graph()
    try:
        g.parse(file_path, format="turtle")

        # Bind common namespaces after parsing with override=True to ensure deterministic serialization
        g.namespace_manager.bind(
            "dc", Namespace("http://purl.org/dc/terms/"), override=True, replace=True
        )
        g.namespace_manager.bind(
            "prov", Namespace("http://www.w3.org/ns/prov#"), override=True, replace=True
        )
        g.namespace_manager.bind(
            "bfo",
            Namespace("http://purl.obolibrary.org/obo/BFO_"),
            override=True,
            replace=True,
        )
        g.namespace_manager.bind(
            "schema", Namespace("http://schema.org/"), override=True, replace=True
        )
        g.namespace_manager.bind(
            "owl",
            Namespace("http://www.w3.org/2002/07/owl#"),
            override=True,
            replace=True,
        )
        g.namespace_manager.bind(
            "rdfs",
            Namespace("http://www.w3.org/2000/01/rdf-schema#"),
            override=True,
            replace=True,
        )
        g.namespace_manager.bind(
            "rdf",
            Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            override=True,
            replace=True,
        )
        g.namespace_manager.bind(
            "xsd",
            Namespace("http://www.w3.org/2001/XMLSchema#"),
            override=True,
            replace=True,
        )

        serialized = g.serialize(format="turtle")

        # Ensure standard prefixes exist in the header if rdflib stripped them
        required_prefixes = {
            "dc": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "time": "http://www.w3.org/2006/time#",
            "bibo": "http://purl.org/ontology/bibo/",
            "fibo-org": "https://spec.edmcouncil.org/fibo/ontology/FND/Organizations/Organizations/",
            "fibo-fi": "https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/",
        }
        for prefix, uri in required_prefixes.items():
            if f"@prefix {prefix}:" not in serialized:
                serialized = f"@prefix {prefix}: <{uri}> .\n" + serialized

        with open(file_path, "w") as f:
            f.write(serialized)

        return True
    except Exception as e:
        print(f"Error formatting {file_path}: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = True
    for arg in sys.argv[1:]:
        if not format_ttl(arg):
            success = False
    sys.exit(0 if success else 1)
