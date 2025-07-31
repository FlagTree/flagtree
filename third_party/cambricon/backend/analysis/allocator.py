import sys


def align(val: int, align: int) -> int:
    """
    Aligns `val` to the nearest multiple of `align`.
    If `val` is already a multiple of `align`, returns `val`;
    otherwise, returns the smallest multiple of `align` greater than `val`.
    """
    if align == 0:
        raise ValueError("align must not be 0")
    return (val + align - 1) // align * align


class BFCAllocator:

    def __init__(self):
        """
        Initialize the memory allocator.
        """
        self.memory_pool = []  # Simulate memory pool: record allocated memory blocks (start offset, size)
        self.free_blocks = []  # Free block list: (start offset, size)
        self.mem_pool_size = 0  # Current total size of the memory pool (also the peak memory usage)

    def alloc(self, size):
        """
        Allocate a memory block of the specified size.
        :param size: The requested memory size
        :return: The starting offset of the allocated block, or -1 if allocation fails
        """
        if size < 0:
            raise ValueError("The requested memory size must be greater than 0.")

        # Currently, when there is a tensor with a scalar shape in the IR,
        # a piece of memory with a size of 0 will be allocated.
        # Since the allocator cannot handle this situation for the time being,
        # 0 is forcibly changed to 1 to bypass it.
        if size == 0:
            size = 1

        # Find the best fit free block
        best_fit_index = -1
        best_fit_size = sys.maxsize  # Initialize with a large value

        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size and block_size < best_fit_size:
                # Found a better fit
                best_fit_index = i
                best_fit_size = block_size

        if best_fit_index != -1:
            # Found a suitable free block
            offset, block_size = self.free_blocks.pop(best_fit_index)
            if block_size > size:
                # Add the remaining part back to the free block list
                self.free_blocks.append((offset + size, block_size - size))
            self.memory_pool.append((offset, size))  # Record the allocated memory block
            return offset

        # No suitable free block found, check if the last block is free and can be expanded
        if self.free_blocks:
            # Sort free blocks by offset to find the last block
            self.free_blocks.sort()
            last_offset, last_size = self.free_blocks[-1]
            if last_offset + last_size == self.mem_pool_size:
                # The last block is free and adjacent to the end of the memory pool
                # Expand this block to satisfy the request
                expanded_size = last_size + size
                self.free_blocks[-1] = (last_offset, expanded_size)
                # Allocate from the expanded block
                offset, block_size = self.free_blocks.pop()
                if block_size > size:
                    # Add the remaining part back to the free block list
                    self.free_blocks.append((offset + size, block_size - size))
                self.memory_pool.append((offset, size))  # Record the allocated memory block
                self.mem_pool_size = offset + size  # Update total size if necessary
                return offset

        # No suitable free block found, expand the memory pool
        offset = self.mem_pool_size  # The new allocation starts at the end of the current memory pool
        self.mem_pool_size += size  # Expand the memory pool
        self.memory_pool.append((offset, size))  # Record the allocated memory block
        return offset

    def free(self, offset):
        """
        Free the memory block at the specified offset.
        :param offset: The starting offset of the block to be freed
        """
        # Find the allocated memory block
        free_success = False
        for i, (block_offset, block_size) in enumerate(self.memory_pool):
            if block_offset == offset:
                # Found the block to free
                del self.memory_pool[i]  # Remove from the allocated list
                self.free_blocks.append((block_offset, block_size))  # Add to the free block list
                free_success = True
                break

        if free_success == False:
            raise ValueError(f"Free invalid memory: {offset}.")

        # Merge adjacent free blocks
        self.free_blocks.sort()
        for i, (current_offset, current_size) in enumerate(self.free_blocks):
            if current_offset != offset:
                continue

            mergePrev = False
            if i > 0:
                pre_offset, pre_size = self.free_blocks[i - 1]
                if pre_offset + pre_size == current_offset:
                    current_offset = pre_offset
                    current_size = current_size + pre_size
                    self.free_blocks[i] = (current_offset, current_size)
                    mergePrev = True

            mergeNext = False
            if i < len(self.free_blocks) - 1:
                next_offset, next_size = self.free_blocks[i + 1]
                if current_offset + current_size == next_offset:
                    self.free_blocks[i] = (current_offset, current_size + next_size)
                    mergeNext = True

            if mergeNext:
                del self.free_blocks[i + 1]
            if mergePrev:
                del self.free_blocks[i - 1]
            break

    def total_size(self):
        """
        Return the peak memory usage (i.e., the total size of the memory pool).
        :return: The peak memory usage
        """
        return self.mem_pool_size

    def __str__(self):
        """
        Return a string representation of the current memory state.
        """
        return (f"Total memory pool size: {self.mem_pool_size}, "
                f"Free blocks: {self.free_blocks}, "
                f"Allocated blocks: {self.memory_pool}")
