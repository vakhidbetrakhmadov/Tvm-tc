import time
import argparse
import torch
import tensor_comprehensions as tc

def build_and_time(args: argparse.Namespace, tc_str: str, entry_point: str, *inputs: torch.Tensor) -> tc.Executor:
    start = time.clock()
    exe = build(args, tc_str, entry_point, *inputs)
    end = time.clock()
    print("Done compiling \"{}\" (compile time: {}ms)".format(entry_point, (end - start) * 10 ** 3))
    return exe
        
def build(args: argparse.Namespace, tc_str: str, entry_point: str, *inputs: torch.Tensor) -> tc.Executor:
    tuner_config = (
        tc.TunerConfig()
        .threads(args.tuner_threads)
        .generations(args.tuner_generations)
        .pop_size(args.tuner_pop_size)
        .number_elites(args.tuner_number_elites)
        .devices(args.tuner_devices))

    if args.autotuner:
        if args.debug:  print("Running autotuner.")
        
        if args.load_from_cache:
            return tc.autotune_and_compile(tc_str,
                                       entry_point,
                                       *inputs,
                                       starting_options=None,
                                       tuner_config=tuner_config,
                                       cache_filename=args.tuner_cache_file,
                                       load_from_cache=args.load_from_cache,
                                       store_to_cache=args.store_to_cache)
        else: 
            return tc.autotune_and_compile(tc_str,
                                       entry_point,
                                       *inputs,
                                       starting_options='naive',
                                       tuner_config=tuner_config,
                                       cache_filename=args.tuner_cache_file,
                                       load_from_cache=args.load_from_cache,
                                       store_to_cache=args.store_to_cache)

    elif args.load_from_cache:
        if args.debug:  print("Loading autotuned mapping options from cache.")
            
        mapping_options = tc.make_load_from_cache_options_factory(args.tuner_cache_file)(tc_str, entry_point, *inputs)
        return tc.compile(tc_str, entry_point, mapping_options, *inputs)
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
        if args.useReadOnlyCache is not None:
            options.useReadOnlyCache(args.useReadOnlyCache)
        if args.matchLibraryCalls is not None:
            options.matchLibraryCalls(args.matchLibraryCalls)
        if args.fixParametersBeforeScheduling is not None:
            options.fixParametersBeforeScheduling(args.fixParametersBeforeScheduling)
        if args.outerScheduleFusionStrategy is not None:
            options.outerScheduleFusionStrategy(args.outerScheduleFusionStrategy)
        if args.intraTileScheduleFusionStrategy is not None:
            options.intraTileScheduleFusionStrategy(args.intraTileScheduleFusionStrategy)

        return tc.compile(tc_str, entry_point, options, *inputs)