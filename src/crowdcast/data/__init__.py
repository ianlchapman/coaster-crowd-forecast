"""Input tables: schema contracts, loaders and a synthetic dataset for tests and demos."""

from crowdcast.data.loaders import load_crowd_calendar, load_events, load_parks
from crowdcast.data.schema import SchemaError, TableSpec, validate

__all__ = ["SchemaError", "TableSpec", "load_crowd_calendar", "load_events", "load_parks", "validate"]
