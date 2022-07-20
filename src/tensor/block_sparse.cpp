#include "block_sparse.h"

namespace torque {
namespace block_sparse {

ContractionInfo
block_in_range(const arma::umat & contracting_indices,
               const arma::uvec & A_begin_point,
               const arma::uvec & A_end_point,
               const arma::umat & B_begin_points,
               const arma::umat & B_end_points) {

  const arma::uword B_n_blocks = B_begin_points.n_cols;
  const arma::uvec A_contracting_indices = contracting_indices.col(0);
  const arma::uvec B_contracting_indices = contracting_indices.col(1);

  // assert A block and B block have consistent number of subblocks and rank
  assert(A_begin_point.n_rows == A_end_point.n_rows);
  assert(B_begin_points.n_rows == B_end_points.n_rows);
  assert(B_begin_points.n_cols == B_end_points.n_cols);

  // The intervals of the non-trivial contribution from the blocks are min(end) - max(begin)
  arma::umat max_begin_indices_in_contracting_dimension(
      arma::size(contracting_indices.n_rows,
                 B_n_blocks), arma::fill::zeros);

  arma::umat min_end_indices_in_contracting_dimension(
      arma::size(contracting_indices.n_rows,
                 B_n_blocks), arma::fill::zeros);

  const arma::uvec A_sort_index = arma::sort_index(A_contracting_indices);
  const arma::uvec B_sort_index = arma::sort_index(B_contracting_indices);

  const arma::uvec sorted_A_contracting_indices = A_contracting_indices(
      A_sort_index);
  const arma::uvec sorted_B_contracting_indices = B_contracting_indices(
      B_sort_index);

  // Check whether it has non-trivial intervals for each block
  arma::Col<int> true_false_list(B_n_blocks, arma::fill::zeros);

  for (arma::uword i = 0; i < B_n_blocks; i++) {
    const arma::uvec i_begin_point = B_begin_points.col(
        i); // sub-block from B list
    const arma::uvec i_end_point = B_end_points.col(i);

    const arma::uvec max_begin_indices = arma::max(
        A_begin_point.rows(A_contracting_indices),
        i_begin_point.rows(B_contracting_indices));
    const arma::uvec min_end_indices = arma::min(
        A_end_point.rows(A_contracting_indices),
        i_end_point.rows(B_contracting_indices));

    if (arma::all(max_begin_indices <= min_end_indices)) {
      true_false_list(i) = 1;

      max_begin_indices_in_contracting_dimension.col(i) = max_begin_indices;
      min_end_indices_in_contracting_dimension.col(i) = min_end_indices;
    }
  }

  const arma::uvec non_trivial_block_index = arma::find(true_false_list);

  if (non_trivial_block_index.n_elem) {

    arma::uvec new_begin_point_from_A = A_begin_point;
    new_begin_point_from_A.shed_rows(A_contracting_indices);

    arma::umat new_begin_points_from_B = B_begin_points.cols(
        non_trivial_block_index);
    new_begin_points_from_B.shed_rows(B_contracting_indices);

    const arma::uword new_rank =
        new_begin_point_from_A.n_elem + new_begin_points_from_B.n_rows;

    const arma::umat new_begin_points =
        new_rank ?
        arma::join_vert(arma::repmat(new_begin_point_from_A, 1,
                                     non_trivial_block_index.n_elem),
                        new_begin_points_from_B) :
        arma::umat{};

    arma::umat new_begin_points_for_A =
        arma::repmat(new_begin_point_from_A, 1, non_trivial_block_index.n_elem);

    arma::umat new_begin_points_for_B = new_begin_points_from_B;


    for (arma::uword k = 0; k < A_contracting_indices.n_elem; k++) {
      new_begin_points_for_A.insert_rows(sorted_A_contracting_indices(k), 1);
      new_begin_points_for_B.insert_rows(sorted_B_contracting_indices(k), 1);
    }

    const arma::umat non_trivial_max_begin_indices =
        max_begin_indices_in_contracting_dimension.cols(
            non_trivial_block_index);
    const arma::umat non_trivial_min_end_indices =
        min_end_indices_in_contracting_dimension.cols(non_trivial_block_index);

    for (arma::uword k = 0; k < A_contracting_indices.n_elem; k++) {
      new_begin_points_for_A.row(A_contracting_indices(k)) =
          non_trivial_max_begin_indices.row(k);
      new_begin_points_for_B.row(B_contracting_indices(k)) =
          non_trivial_max_begin_indices.row(k);
    }

    arma::uvec new_end_point_from_A = A_end_point;
    new_end_point_from_A.shed_rows(A_contracting_indices);

    arma::umat new_end_points_from_B = B_end_points.cols(
        non_trivial_block_index);
    new_end_points_from_B.shed_rows(B_contracting_indices);

    const arma::umat new_end_points =
        new_rank ?
        arma::join_vert(arma::repmat(new_end_point_from_A, 1,
                                     non_trivial_block_index.n_elem),
                        new_end_points_from_B) :
        arma::umat{};

    arma::umat contracting_tables(
        arma::size(max_begin_indices_in_contracting_dimension));

    for (arma::uword i = 0; i < non_trivial_block_index.n_elem; i++) {
      contracting_tables.col(i) = util::generate_index_table(
          non_trivial_min_end_indices.col(i)
          - non_trivial_max_begin_indices.col(i)
          + arma::ones<arma::uvec>(contracting_indices.n_rows));

    }

    arma::umat new_end_points_for_A =
        arma::repmat(new_end_point_from_A, 1, non_trivial_block_index.n_elem);

    arma::umat new_end_points_for_B = new_end_points_from_B;


    for (arma::uword k = 0; k < A_contracting_indices.n_elem; k++) {
      new_end_points_for_A.insert_rows(sorted_A_contracting_indices(k), 1);
      new_end_points_for_B.insert_rows(sorted_B_contracting_indices(k), 1);
    }

    for (arma::uword k = 0; k < A_contracting_indices.n_elem; k++) {
      new_end_points_for_A.row(A_contracting_indices(k)) =
          non_trivial_min_end_indices.row(k);
      new_end_points_for_B.row(B_contracting_indices(k)) =
          non_trivial_min_end_indices.row(k);
    }

    return {non_trivial_block_index,
            new_begin_points,
            new_end_points,
            non_trivial_max_begin_indices,
            non_trivial_min_end_indices,
            contracting_tables,
            new_begin_points_for_A,
            new_end_points_for_A,
            new_begin_points_for_B,
            new_end_points_for_B
    };

  } else {
    return {};
  }

}

}
}