// Both shims are injected only into the validation script's owned process.
#include "folder-panel.m"

static NSArray<NSURL *> *open_urls(NSOpenPanel *panel, SEL selector) {
    return selected ? @[selected] : @[];
}

static NSURL *save_selection;
static NSInteger save_answer(NSSavePanel *panel, SEL selector) {
    @autoreleasepool {
        const char *path = getenv("BREEZE_SAVE_TEST_PATH");
        save_selection = path ? [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] : nil;
        fprintf(stderr, "[save-test] selected=%s\n", path ?: "cancel");
        fflush(stderr);
        return path ? NSModalResponseOK : NSModalResponseCancel;
    }
}
static NSURL *saved_url(NSSavePanel *panel, SEL selector) { return save_selection; }
__attribute__((constructor)) static void install_save(void) {
    if (!getenv("R2T2_PICKER_TEST_PLAN")) return;
    class_replaceMethod(NSOpenPanel.class, @selector(URLs), (IMP)open_urls,
                        method_getTypeEncoding(class_getInstanceMethod(NSOpenPanel.class, @selector(URLs))));
    Class cls = NSSavePanel.class;
    class_replaceMethod(cls, @selector(runModal), (IMP)save_answer,
                        method_getTypeEncoding(class_getInstanceMethod(cls, @selector(runModal))));
    class_replaceMethod(cls, @selector(URL), (IMP)saved_url,
                        method_getTypeEncoding(class_getInstanceMethod(cls, @selector(URL))));
}
