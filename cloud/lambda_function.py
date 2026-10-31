"""AWS Lambda entry point: handler = lambda_function.handler.

Event: {"framework": "langgraph" | "strands", "steps": 40, "replicates": 30}
"""
import bench


def handler(event, context):
    return bench.run(**(event or {}))
