"""Persistent GPU worker; stdin jobs and stdout JSON replies are private to the sweep."""

import json
import os
import sys
import traceback


def main():
    # Keep protocol output separate from Python and native-library diagnostics.
    reply = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    client = None
    try:
        for line in sys.stdin:
            job = json.loads(line)
            sys.stdout.flush()
            sys.stderr.flush()
            with open(job['log'], 'a') as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                try:
                    import logging
                    import tyro
                    from benchmark.client import LocalClientPolicy, parse_client_args
                    from benchmark.libero import Args, _eval_libero

                    logging.basicConfig(level=logging.INFO)

                    def evaluate(args: Args, client_args: str = '{}'):
                        nonlocal client
                        _, options = parse_client_args(client_args)
                        if client is None:
                            client = LocalClientPolicy(**options)
                        else:
                            client.configure_run(num_steps=options['num_steps'],
                                                 encode_keep_rate=options['encode_keep_rate'],
                                                 debug_dir=options['debug_dir'])
                        _eval_libero(args, client_args, client=client)

                    tyro.cli(evaluate, args=job['argv'])
                    code = 0
                except (Exception, SystemExit):
                    traceback.print_exc()
                    # A failed CUDA operation may poison the context. Let the parent
                    # start a fresh process for the next configuration.
                    code = 1
                sys.stdout.flush()
                sys.stderr.flush()
                reply.write(json.dumps({'returncode': code}) + '\n')
                if code:
                    return
    finally:
        if client is not None:
            client.close()
        reply.close()


if __name__ == '__main__':
    main()
