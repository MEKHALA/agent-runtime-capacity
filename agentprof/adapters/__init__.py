"""Adapters hook a real framework's call sites into the Tracer.

These cannot be exercised here without the frameworks installed, so they are
deliberately thin: each one only has to open the right SpanKind around the
right call. Test each against its framework before you trust its numbers.
"""
