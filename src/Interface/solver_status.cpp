#include "Interface/solver_status.hpp"

namespace ippmm {

    const char* to_string(SolveStatus s) {
        switch (s) {
            case SolveStatus::Optimal:              return "optimal";
            case SolveStatus::MaxIterations:        return "max iterations reached";
            case SolveStatus::PrimalInfeasible:     return "primal infeasible";
            case SolveStatus::DualInfeasible:       return "dual infeasible";
            case SolveStatus::NumericalError:       return "numerical error (ill-conditioning)";
            case SolveStatus::InsufficientAccuracy: return "insufficient accuracy";
        }
        return "unknown";
    }

}  