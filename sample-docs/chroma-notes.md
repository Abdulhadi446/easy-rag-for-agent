# Chroma notes

Chroma is an open-source embedding database. It stores vectors next to metadata and
documents, so a single query returns both the match and the original text.

## Collections

A collection is a named group of embeddings. Names must be 3-512 characters and may
contain letters, digits, dots, dashes and underscores.

## Distance

Chroma supports l2 and cosine distance. easy-rag uses cosine distance, where a lower
score means a closer match.

## Persistence

PersistentClient writes to a SQLite file plus a segment directory, so reopening the
same path restores the collection.
