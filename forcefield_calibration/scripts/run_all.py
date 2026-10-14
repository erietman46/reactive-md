"""Run the complete pre-MD calibration pipeline (about 20 minutes on four cores; Stage 1 runs the families in parallel)."""
import run_export
import run_stage0_reference
import run_stage1_generic
import run_stage1_pair
import run_stage2_manybody
import run_stage3_selection
import make_figures  # noqa: F401  (configures matplotlib on import)

if __name__ == "__main__":
    run_stage0_reference.main()
    run_stage1_pair.main()
    run_stage1_generic.main()
    run_stage2_manybody.main()
    run_stage3_selection.main()
    run_export.main()
    make_figures.main()
