define <1024 x i32> @hydride.node.tensor_add_bitserial_depth2_rafae.0(<1024 x i32> %arg, <1024 x i32> %arg.1) local_unnamed_addr {
entry:
  %pimAlloc = tail call i32 @pimAlloc(i32 0, i64 1024, i32 2)
  %0 = alloca <1024 x i32>, align 4096
  store <1024 x i32> %arg, <1024 x i32>* %0, align 4096
  %1 = bitcast <1024 x i32>* %0 to i8*
  %pimHostToDevice = call i32 @pimCopyHostToDevice(i8* nonnull %1, i32 %pimAlloc, i64 0, i64 0)
  %pimAssocAlloc = call i32 @pimAllocAssociated(i32 %pimAlloc, i32 2)
  %2 = alloca <1024 x i32>, align 4096
  store <1024 x i32> %arg.1, <1024 x i32>* %2, align 4096
  %3 = bitcast <1024 x i32>* %2 to i8*
  %pimHostToDevice1 = call i32 @pimCopyHostToDevice(i8* nonnull %3, i32 %pimAssocAlloc, i64 0, i64 0)
  %pimAssocAlloc2 = call i32 @pimAllocAssociated(i32 %pimAlloc, i32 2)
  %pimInst = call i32 @pimAdd(i32 %pimAlloc, i32 %pimAssocAlloc, i32 %pimAssocAlloc2)
  %4 = alloca <1024 x i32>, align 4096
  %5 = bitcast <1024 x i32>* %4 to i8*
  %pimDeviceToHost = call i32 @pimCopyDeviceToHost(i32 %pimAssocAlloc2, i8* nonnull %5, i64 0, i64 0)
  %load.buffer = load <1024 x i32>, <1024 x i32>* %4, align 4096
  %pimFree = call i32 @pimFree(i32 %pimAssocAlloc2)
  %pimFree3 = call i32 @pimFree(i32 %pimAlloc)
  %pimFree4 = call i32 @pimFree(i32 %pimAssocAlloc)
  ret <1024 x i32> %load.buffer
}