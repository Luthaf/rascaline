/// Types that can be used as the backing store for a `StackVec`
pub unsafe trait Array: Default {
    /// The type of the array's elements.
    type Item;
    /// Returns the number of items the array can hold.
    fn capacity() -> usize;
    /// Returns a pointer to the first element of the array
    fn ptr(&self) -> *const Self::Item;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            unsafe impl<T: Default> Array for [T; $size] {
                type Item = T;
                fn capacity() -> usize { $size }
                fn ptr(&self) -> *const T {
                    &self[0] as *const _
                }
            }
        )+
    }
);
impl_array!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
);


/// A stack-allocated vector with fixed capacity. If more elements than the
/// initial capacity are needed, this will panic.
///
/// We use this in rascaline instead of
/// [smallvec](https://crates.io/crates/smallvec) because removing the branching
/// between stack & heap allocation improves performance for us.
pub struct StackVec<A: Array> {
    /// The current number of elements stored in the array
    len: usize,
    /// The array used for storage
    data: A
}

impl<A: Array> StackVec<A> {
    /// Create a new empty `StackVec`. The capacity is controlled by the type
    /// parameter `A`
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            len: 0,
            data: Default::default(),
        }
    }

    /// Add an element to this `StackVec`
    ///
    /// # Panics
    ///
    /// If the backing array is already full
    pub fn push(&mut self, value: A::Item) {
        if self.len == A::capacity() {
            panic!("StackVec is full, use a larger array as storage")
        }

        // SAFETY: we just checked that we are in bounds
        unsafe {
            let ptr = self.data.ptr().add(self.len) as *mut A::Item;
            ptr.write(value);
        }

        self.len += 1;
    }
}

impl<A: Array> std::ops::Deref for StackVec<A> {
    type Target = [A::Item];

    fn deref(&self) -> &Self::Target {
        // SAFETY: self.len is always smaller than the size of the array, and
        // all values in the array are valid because we initialize all elements
        // to `Default`
        unsafe {
            std::slice::from_raw_parts(self.data.ptr(), self.len)
        }
    }
}
