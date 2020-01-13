#!/usr/bin/env python
"""
Sends an alert when a ship stops multiple time close to an EEZ
"""

import argparse
from collections import defaultdict

from sesamelib.faust import MutatedDynamicMessage
from sesamelib.multitask import Trajectory

import faust


APP_NAME = "multitaskais_stream"

# 4 hours (in seconds)
# MAX_TIMESPAN = 4 * 60 * 60
MAX_TIMESPAN = 300

TRAJECTORIES = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Alert if a ship enters one of the indexed EEZs")
    parser.add_argument("--bootstrap_servers", "-b",
                        help="Kafka bootstrap servers",
                        action="append",
                        required=True)
    args = parser.parse_args()

    app = faust.App(
        APP_NAME,
        broker="kafka://{}".format(",".join(args.bootstrap_servers))
    )

    topic = app.topic("ais.dynamic", key_type=str, value_type=MutatedDynamicMessage)
    out_topic = app.topic(APP_NAME)

    channel = app.channel(value_type=str)

    @app.agent(channel)
    async def process(stream):
        async for event in stream:
            print(f'Received: {event!r}')
            # call a contrario detection here

    @app.agent(topic)
    async def mutate_all(msgs):
        async for msg in msgs:
            TRAJECTORIES.setdefault(msg.mmsi, Trajectory(MAX_TIMESPAN))
            current_traj = TRAJECTORIES[msg.mmsi]
            current_traj.add(msg)
            if current_traj.timespan() > MAX_TIMESPAN:
               await channel.send(value=(msg.mmsi, current_traj.timespan()))

    app.finalize()
    worker = faust.Worker(app,
                          loglevel="INFO")

    worker.execute_from_commandline()