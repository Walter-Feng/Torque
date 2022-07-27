#include "block_sparse.h"

namespace torque {
namespace block_sparse {

arma::uvec sort_index_helper(const arma::uword length,
                             const arma::uvec & contracting_indices) {
  arma::uvec full(length);
  for (arma::uword i = 0; i < length; i++) {
    full(i) = i;
  }

  full.shed_rows(contracting_indices);

  return arma::sort_index(arma::join_vert(full, contracting_indices));
}

arma::umat glue_back_contraction_dimension(arma::umat original,
                                           const arma::umat & contraction,
                                           const arma::uvec & contracting_indices) {
  assert(original.n_cols == contraction.n_cols);
  original.shed_rows(contracting_indices);

  const arma::umat glued = arma::join_vert(original, contraction);

  return glued.rows(sort_index_helper(glued.n_rows, contracting_indices));
}

template<bool is_max>
arma::ucube
get_extremum(const arma::umat & A,
             const arma::umat & B) {
  assert(A.n_rows == B.n_rows);

  arma::umat A_rep = arma::repmat(A, 1, B.n_cols);
  arma::umat B_rep = arma::repelem(B, 1, A.n_cols);

  arma::ucube A_rep_in_cube(A_rep.memptr(), A.n_rows, A.n_cols, B.n_cols,
                            false);
  arma::ucube B_rep_in_cube(B_rep.memptr(), A.n_rows, A.n_cols, B.n_cols,
                            false);

  if constexpr(is_max) {
    return arma::max(A_rep_in_cube, B_rep_in_cube);
  } else {
    return arma::min(A_rep_in_cube, B_rep_in_cube);
  }

}

reducedContractionInfo
optimized_block_in_range(const arma::umat & contracting_indices,
                         const arma::umat & A_begin_points,
                         const arma::umat & A_end_points,
                         const arma::umat & B_begin_points,
                         const arma::umat & B_end_points) {

  const arma::uvec A_contracting_indices = contracting_indices.col(0);
  const arma::uvec B_contracting_indices = contracting_indices.col(1);

  const arma::umat A_begin_points_contraction = A_begin_points.rows(
      A_contracting_indices);
  const arma::umat A_end_points_contraction = A_end_points.rows(
      A_contracting_indices);

  const arma::umat B_begin_points_contraction = B_begin_points.rows(
      B_contracting_indices);
  const arma::umat B_end_points_contraction = B_end_points.rows(
      B_contracting_indices);

  arma::ucube begin_points_max = get_extremum<true>(
      A_begin_points_contraction, B_begin_points_contraction);

  arma::ucube end_points_min = get_extremum<false>(
      A_end_points_contraction, B_end_points_contraction);

  const arma::uvec valid_indices = arma::find(
      begin_points_max < end_points_min);

  arma::Mat<int> valid_indices_mask(begin_points_max.n_rows,
                                    begin_points_max.n_cols *
                                    begin_points_max.n_slices,
                                    arma::fill::zeros);

  valid_indices_mask(valid_indices) += 1;

  const arma::uvec non_trivial_indices = arma::find(
      arma::prod(valid_indices_mask));

  arma::umat begin_points_max_in_mat(begin_points_max.memptr(),
                                     begin_points_max.n_rows,
                                     begin_points_max.n_cols *
                                     begin_points_max.n_slices, false);

  arma::umat end_points_min_in_mat(end_points_min.memptr(),
                                   end_points_min.n_rows,
                                   end_points_min.n_cols *
                                   end_points_min.n_slices, false);

  const arma::umat non_trivial_block_indices = arma::ind2sub(
      arma::size(A_begin_points.n_cols, B_begin_points.n_cols),
      non_trivial_indices);

  const auto glued_A_begin_points =
      glue_back_contraction_dimension(
          A_begin_points.cols(non_trivial_block_indices.row(0).t()),
          begin_points_max_in_mat.cols(non_trivial_indices), A_contracting_indices);

  const auto glued_A_end_points =
      glue_back_contraction_dimension(
          A_end_points.cols(non_trivial_block_indices.row(0).t()),
          end_points_min_in_mat.cols(non_trivial_indices), A_contracting_indices);

  const auto glued_B_begin_points =
      glue_back_contraction_dimension(
          B_begin_points.cols(non_trivial_block_indices.row(1).t()),
          begin_points_max_in_mat.cols(non_trivial_indices), B_contracting_indices);

  const auto glued_B_end_points =
      glue_back_contraction_dimension(
          B_end_points.cols(non_trivial_block_indices.row(1).t()),
          end_points_min_in_mat.cols(non_trivial_indices), B_contracting_indices);

  arma::umat glued_A_begin_points_copy = glued_A_begin_points;
  glued_A_begin_points_copy.shed_rows(A_contracting_indices);
  arma::umat glued_A_end_points_copy = glued_A_end_points;
  glued_A_end_points_copy.shed_rows(A_contracting_indices);

  arma::umat glued_B_begin_points_copy = glued_B_begin_points;
  glued_B_begin_points_copy.shed_rows(B_contracting_indices);
  arma::umat glued_B_end_points_copy = glued_B_end_points;
  glued_B_end_points_copy.shed_rows(B_contracting_indices);

  return {
    non_trivial_block_indices,
    arma::join_vert(glued_A_begin_points_copy, glued_B_begin_points_copy),
    arma::join_vert(glued_A_end_points_copy, glued_B_end_points_copy),
    glued_A_begin_points,
    glued_A_end_points,
    glued_B_begin_points,
    glued_B_end_points
  };

}


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
          + 1);

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