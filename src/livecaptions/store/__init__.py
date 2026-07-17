"""Transcript persistence: save every finalized caption, search it later.

SQLite in WAL mode with an FTS5 full-text index. Writes happen on a dedicated
writer thread with batched inserts — never on the audio or GUI thread.
"""
