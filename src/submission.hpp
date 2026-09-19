#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
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
void apply_stencil(const Grid &old_grid, Grid &new_grid)
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

    // The reason I put __restrict is because old_grid and new_grid never overlap in memory.
    // Without __restrict, the compiler would conservatively assume that writing through a
    // pointer like middle_row_new could affect what gets read through a pointer like
    // middle_row_old, even though old_grid and new_grid are guaranteed to be separate
    // in memory. Therefore, __restrict unlocks aliasing optimizations the compiler
    // couldn't otherwise make.

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