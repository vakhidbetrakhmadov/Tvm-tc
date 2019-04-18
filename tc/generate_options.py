import argparse
import torch
import tensor_comprehensions as tc

def generate_options(args: argparse.Namespace):

    tuner_config = (
        tc.TunerConfig()
        .threads(args.tuner_threads)
        .generations(args.tuner_generations)
        .pop_size(args.tuner_pop_size)
        .number_elites(args.tuner_number_elites)
        .devices(args.tuner_devices))

    def _generate_options(tc_str: str,
                        entry_point: str,
                        *inputs: torch.Tensor) -> tc.MappingOptions:
        global reinforce

        if entry_point == 'make_idx':
            return tc.make_naive_options_factory()(tc_str, entry_point, *inputs)

        if args.autotuner:
            if args.debug:  print("Running autotuner.")

            loaded = tc.make_load_from_cache_options_factory(args.tuner_cache_file)(tc_str, entry_point, *inputs) if args.reinforce else None
            start = loaded if loaded is not None else 'naive'

            return tc.make_autotuned_options_factory(
                starting_options=start,
                tuner_config=tuner_config,
                cache_filename=args.tuner_cache_file,
                store_to_cache=True,)(tc_str, entry_point, *inputs)
        else: 
            if args.debug: print("Building mapping options.")

            options = tc.MappingOptions("naive")

            if args.mapToBlocks is not None:
                options.mapToBlocks(args.mapToBlocks)
            if args.mapToThreads is not None:
                options.mapToThreads(args.mapToThreads)
            if args.tile is not None:
                options.tile(args.tile)
            if args.useSharedMemory is not None:
                options.useSharedMemory(args.useSharedMemory)
            if args.maxSharedMemory is not None:
                options.maxSharedMemory(args.maxSharedMemory)
            if args.unroll is not None:
                options.unroll(args.unroll)
            if args.unrollCopyShared is not None:
                options.unrollCopyShared(args.unrollCopyShared)
            if args.useReaOnlyCache is not None:
                options.useReaOnlyCache(args.useReaOnlyCache)
            if args.matchLibraryCalls is not None:
                options.matchLibraryCalls(args.matchLibraryCalls)
            if args.fixParametersBeforeScheduling is not None:
                options.fixParametersBeforeScheduling(args.fixParametersBeforeScheduling)
            if args.outerScheduleFusionStrategy is not None:
                options.outerScheduleFusionStrategy(args.outerScheduleFusionStrategy)
            if args.intraTileFusionStrategy is not None:
                options.intraTileFusionStrategy(args.intraTileFusionStrategy)
            
            return options
            
    return _generate_options