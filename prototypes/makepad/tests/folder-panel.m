// Test-only NSOpenPanel/NSSavePanel responses for an owned hidden process. The production
// executable never links this shim; DYLD_INSERT_LIBRARIES is set only by the test.
// Exercises Makepad's real native dialog callback without opening a modal window
// on the user's desktop. It does not test mouse/keyboard operation of NSOpenPanel.
#import <AppKit/AppKit.h>
#import <objc/runtime.h>

static NSURL *selected;
static NSUInteger response_index;

static NSInteger answer(NSSavePanel *panel, SEL selector) {
    @autoreleasepool {
        NSString *planPath = [NSString stringWithUTF8String:getenv("R2T2_PICKER_TEST_PLAN")];
        NSArray *plan = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:planPath]
                                                       options:0 error:nil];
        id choice = response_index < plan.count ? plan[response_index] : NSNull.null;
        NSDictionary *record = @{@"index": @(response_index++), @"choice": choice,
                                 @"message": panel.message ?: @"",
                                 @"location": panel.directoryURL.path ?: @""};
        NSData *json = [NSJSONSerialization dataWithJSONObject:record options:0 error:nil];
        fprintf(stderr, "[picker-test] %s\n", [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        fflush(stderr);
        selected = [choice isKindOfClass:NSString.class] ? [NSURL fileURLWithPath:choice] : nil;
        return selected ? NSModalResponseOK : NSModalResponseCancel;
    }
}

static NSURL *selection(NSSavePanel *panel, SEL selector) {
    return selected;
}

__attribute__((constructor)) static void install(void) {
    if (!getenv("R2T2_PICKER_TEST_PLAN")) return;
    for (Class cls in @[NSSavePanel.class, NSOpenPanel.class]) {
        class_replaceMethod(cls, @selector(runModal), (IMP)answer,
                            method_getTypeEncoding(class_getInstanceMethod(cls, @selector(runModal))));
        class_replaceMethod(cls, @selector(URL), (IMP)selection,
                            method_getTypeEncoding(class_getInstanceMethod(cls, @selector(URL))));
    }
}
