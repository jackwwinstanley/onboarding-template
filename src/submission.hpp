#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

// Stride, width, and alignment:

// The stride between rows each is equal to the width. I decided not to add gaps between rows.
// Since there are 1024 columns, (verified in main.cpp),
// this is a clean benchmark, as 1024 cols * 8 bytes = 8192 bytes, a number divisible by 64.
// However, if I received an odd number, for example, it's impossible to get a number
// divisible by 64.

// Padding:

// I decided not to add padding, because this grid is 1024x1024, it's unnecessary.
// Implementing padding could jeopardize the code infrastructure, as I would have to replace
// std::vector, which comes with safe zero-initialization, safe handling of copies, and
// automatic cleanup. If I implemented padding, it would increase the risk of bugs in grids with
// different column counts (ex: 4x8,16x2), with zero benefit because the only grid tested for
// speed is 1024x1024.

// SIMD and Vectorization:

// I decided not to add SIMD intrinsics, as adding it is unnecessary. This is because
// testing by threading showed that my slowdown was due to memory access, not because of needing faster calculations.
// Additionally, I confirmed that vectorization actually was occurring using the compiler command
// clang++ -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
// verifying that vectorization was occurring in my cols loop. (~line 105)
// Writing another solution wouldn't actually help me in a significant way.

// Parallelization and scale threading:

// Each time I doubled my thread count, the improvement I saw got smaller each time.
// By the time I got to 10 threads, I was experiencing diminished returns because my
// MacBook Pro only has 8 performance cores and 2 efficiency cores. This means when I set my thread
// count to 10, it would evenly distribute the work across all 10 of the cores, including
// the 2 efficiency cores. This takes longer than using only 8 performance cores because
// the performance cores finish their work quickly and then have to wait for the slower efficiency
// cores to finish before producing a final result, which increases overall runtime
// when compared to using 8 performance cores only. This is a result of my machine, and might not
// necessarily apply to the evaluator's machine.

// the Grid class owns and manages the memory.

class Grid
{
private:
  std::size_t rows_;
  std::size_t cols_;

  // grid variable stores all values in rows and cols in 1D vector. Can access individual index
  // by using the formula idx = (row_index * number_of_columns) + cols_index

  std::vector<double> grid;

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), grid(rows * cols, 0.0)
  {
  }

  double &
  operator()(std::size_t i, std::size_t j)
  {
    return grid[(i * cols_) + j];
  }
  double operator()(std::size_t i, std::size_t j) const
  {
    return grid[(i * cols_) + j];
  }

  std::size_t rows() const { return rows_; }

  std::size_t cols() const { return cols_; }

  // Gives access directly to memory for the first element in grid
  // Exists to bulk copy data adjacent in memory from old_grid to new_grid
  double *data() { return &grid[0]; }
  const double *data() const { return &grid[0]; }
};

// The structs View and ConstView are just temporary ways of accessing the memory in the
// Grid class without owning any information.

// Additionally, I decided that each struct would have its own copy of rows and cols.
// This is because this abstraction is simpler and easier, rather than having 3 structs, one accessing
// a const pointer to old_grid, another having a pointer to new_grid, and the third storing the dimensions.
// It also reduces dependency, as neither struct is reliant on the other to get the number of rows
// and cols.

// After introducing the structs View and ConstView (before I just used variables to get
// the pointers, rows, and cols from the Grid class), I reran
// clang++ -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -I src bench/main.cpp -o benchmark
// and got the same results, vectorized width 2, interleaved count 4 for the interior stencil loop,
// and vectorized, width 2, interleaved count 2 for the column boundary loop.

struct View
{
  double *new_grid_ptr;
  std::size_t rows;
  std::size_t cols;
};

struct ConstView
{
  const double *old_grid_ptr;
  const std::size_t rows;
  const std::size_t cols;
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid &old_grid, Grid &new_grid)
{

  ConstView old_grid_info{old_grid.data(), old_grid.rows(), old_grid.cols()};
  View new_grid_info{new_grid.data(), new_grid.rows(), new_grid.cols()};

  // Copies old_grid top row and bottom row to new_grid using memory blocks.
  // This is more efficient than using a for loop because instead of iterating through each item, the
  // vector's values can be instantly replicated onto the new_grid
  std::copy(old_grid_info.old_grid_ptr, old_grid_info.old_grid_ptr + old_grid_info.cols, new_grid_info.new_grid_ptr);                                                                                                                            // Top row
  std::copy(old_grid_info.old_grid_ptr + ((old_grid_info.rows - 1) * old_grid_info.cols), old_grid_info.old_grid_ptr + (old_grid_info.rows * old_grid_info.cols), new_grid_info.new_grid_ptr + ((old_grid_info.rows - 1) * old_grid_info.cols)); // Bottom row

  // Copies the right and leftmost columns from old_grid to new_grid
  // There was no other way to optimize this other than by using for loops. This is because,
  // unlike rows, columns do not sit right next to each other in memory. My configuration of
  // the 1D vector makes it so that everything in a row is right next to each other, but things in
  // different columns are varying distances apart. Therefore, the most
  // efficient solution was to implement for loops.
  for (std::size_t r{0}; r < old_grid_info.rows; ++r)
  {
    new_grid(r, 0) = old_grid(r, 0);
    new_grid(r, old_grid_info.cols - 1) = old_grid(r, old_grid_info.cols - 1);
  }

// Parallelizing the task of applying the stencil to the interior
// rows and columns in each run of the outer for loop.
// This is safe to do because each of these tasks are completely independent of each other.
// Each iteration only reads from an unchanging input and writes a completely distinct output.
#pragma omp parallel for
  for (std::size_t i = 1; i < old_grid_info.rows - 1; ++i)
  {
    // Upon checking the test harness, I can confirm that old_grid and new_grid are completely
    // different objects.
    // As a result, I used __restrict because old_grid and new_grid never overlap in memory.
    // Without __restrict, the compiler would conservatively assume that writing through a
    // pointer like middle_row_new could affect what gets read through a pointer like
    // middle_row_old, even though old_grid and new_grid are guaranteed to be separate
    // in memory. Therefore, __restrict unlocks aliasing optimizations the compiler
    // couldn't otherwise make.

    // Memory alignment:

    // Additionally, the middle_row, above_row, and below_row for old_grid are all aligned.
    // This is because they are only shifted by one row each. However, the ones to the left
    // and right aren't in alignment, as they are shifted one col to the left and right.

    const double *__restrict above_row_old{old_grid_info.old_grid_ptr + old_grid_info.cols * (i - 1)};
    const double *__restrict middle_row_old{old_grid_info.old_grid_ptr + old_grid_info.cols * (i)};
    const double *__restrict below_row_old{old_grid_info.old_grid_ptr + old_grid_info.cols * (i + 1)};
    double *__restrict middle_row_new{new_grid_info.new_grid_ptr + old_grid_info.cols * i};

    for (std::size_t j = 1; j < old_grid_info.cols - 1; ++j)
    {
      middle_row_new[j] = 0.5 * middle_row_old[j] + 0.125 * (above_row_old[j] + below_row_old[j] + middle_row_old[j - 1] + middle_row_old[j + 1]);
    }
  }
}