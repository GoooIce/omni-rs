// Every open, folder and save response is selected by the owned test plan.
#include "folder-panel.m"
static NSArray<NSURL *> *music_open_urls(NSOpenPanel *panel, SEL selector) {
    return selected ? @[selected] : @[];
}
__attribute__((constructor)) static void install_music_open(void) {
    if (!getenv("R2T2_PICKER_TEST_PLAN")) return;
    class_replaceMethod(NSOpenPanel.class, @selector(URLs), (IMP)music_open_urls,
                        method_getTypeEncoding(class_getInstanceMethod(NSOpenPanel.class, @selector(URLs))));
}
