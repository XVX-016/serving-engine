// cupti_probe.cpp - Minimal CUPTI build probe.
// If this compiles and links, CUPTI is usable for activity tracing.
#include <cupti_activity.h>
#include <stdio.h>
int main() {
    CUptiResult r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    printf("cuptiActivityEnable returned: %d\n", (int)r);
    return 0;
}
