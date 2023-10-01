/// Called when memory growth is necessary. Returns a capacity larger than
/// minimum that grows super-linearly.
pub fn growCapacity(current: usize, minimum: usize) usize {
    var new = current;
    while (true) {
        new +|= new / 2 + 8;
        if (new >= minimum)
            return new;
    }
}
