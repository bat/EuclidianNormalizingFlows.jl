    # think about storage, so memory coalescence can be used?, also shared memory caching for older hardware
    # keep data on gpu
    # use profiling to really get in there with the optimization
    #use NVIDIAs compute sanitizer to look at emory issues: cuda/compute-sanitizer
    # instpection tools : @macroexpand use in front of @kernel in kernel definition; @ka_code_typed, @ka_code_llvm use in front of call to kernel