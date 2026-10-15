// Native panel callbacks for an owned hidden image-validation process only.
#import <AppKit/AppKit.h>
#import <objc/runtime.h>
static NSArray<NSURL *> *selected_images;
static NSURL *saved_image;
static NSUInteger open_index,save_index;
static id choice(const char *key,NSUInteger index) {
    const char *path=getenv(key);
    if (!path) return NSNull.null;
    NSArray *plan=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]] options:0 error:nil];
    return index<plan.count ? plan[index] : NSNull.null;
}
static NSInteger open_answer(NSOpenPanel *panel,SEL selector) {
    id value=choice("R2T2_PICKER_TEST_PLAN",open_index++);
    NSArray *paths=[value isKindOfClass:NSArray.class] ? value : ([value isKindOfClass:NSString.class] ? @[value] : @[]);
    NSMutableArray *urls=[NSMutableArray array];
    for (NSString *path in paths) [urls addObject:[NSURL fileURLWithPath:path]];
    selected_images=urls;
    fprintf(stderr,"[image-picker-test] index=%lu count=%lu\n",(unsigned long)(open_index-1),(unsigned long)urls.count);
    fflush(stderr);
    return urls.count ? NSModalResponseOK : NSModalResponseCancel;
}
static NSArray *open_urls(NSOpenPanel *p,SEL s) {return selected_images ?: @[];}
static NSURL *open_url(NSOpenPanel *p,SEL s) {return selected_images.firstObject;}
static NSInteger save_answer(NSSavePanel *p,SEL s) {
    id value=choice("QWEN_IMAGE_SAVE_TEST_PLAN",save_index++);
    saved_image=[value isKindOfClass:NSString.class] ? [NSURL fileURLWithPath:value] : nil;
    return saved_image ? NSModalResponseOK : NSModalResponseCancel;
}
static NSURL *save_url(NSSavePanel *p,SEL s) {return saved_image;}
static void replace(Class cls,SEL sel,IMP fn) {
    class_replaceMethod(cls,sel,fn,method_getTypeEncoding(class_getInstanceMethod(cls,sel)));
}
__attribute__((constructor)) static void install(void) {
    if (!getenv("R2T2_PICKER_TEST_PLAN")) return;
    replace(NSOpenPanel.class,@selector(runModal),(IMP)open_answer);
    replace(NSOpenPanel.class,@selector(URLs),(IMP)open_urls);
    replace(NSOpenPanel.class,@selector(URL),(IMP)open_url);
    replace(NSSavePanel.class,@selector(runModal),(IMP)save_answer);
    replace(NSSavePanel.class,@selector(URL),(IMP)save_url);
}
