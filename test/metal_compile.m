// metal_compile.m (macOS, phase 0): compile a .metal source (e.g. compile_metal's runner.metal) with
// newLibraryWithSource and uirmetal.m's options, and print all of Metal's diagnostics and the time.
//   cc -fobjc-arc -O1 -o /tmp/metal_compile test/metal_compile.m -framework Metal -framework Foundation
//   /tmp/metal_compile DIR/runner.metal
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: mslc FILE.metal\n"); return 2; }
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
        if (!src) { fprintf(stderr, "read: %s\n", error.localizedDescription.UTF8String); return 2; }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 2; }
        MTLCompileOptions *o = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            o.mathMode = MTLMathModeSafe;
            o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            o.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSDate *t = [NSDate date];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:o error:&error];
        double dt = -[t timeIntervalSinceNow];
        if (error) printf("%s\n", error.localizedDescription.UTF8String);
        if (!lib) { printf("FAILED (%.2f s)\n", dt); return 1; }
        printf("OK (%.2f s) on %s; functions: %s\n", dt, device.name.UTF8String,
               [[lib.functionNames componentsJoinedByString:@" "] UTF8String]);
        return 0;
    }
}
