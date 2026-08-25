package strobl.control;

import java.util.Arrays;

public final class ControlledModelTest {
    private static final String[] FAMILIES = new String[] {
        "random_mixed",
        "resistant_core",
        "resistant_edge",
        "resistant_dispersed",
        "two_resistant_nests"
    };

    private ControlledModelTest() {
    }

    public static void main(String[] args) {
        testInitialConditionFamilies();
        testMatchedOccupiedMask();
        testDeterminism();
        testNoMovement();
        testNoFluxAndBlockedDiagnostics();
        testSensitiveOnlyDrugKill();
        testNaturalDeathDiagnostics();
        testAdaptiveBoundary();
        System.out.println("ControlledModelTest: PASS");
    }

    private static ControlledModel model(
            int width, int height, double divisionS, double divisionR,
            double deathS, double deathR, double drugKill) {
        return new ControlledModel(
                width, height, 1.0, divisionS, divisionR,
                deathS, deathR, drugKill);
    }

    private static void testInitialConditionFamilies() {
        for (String family : FAMILIES) {
            ControlledModel model = model(20, 20, 0.1, 0.08, 0.01, 0.01, 0.75);
            model.reset(family, 73, 17, 11L, 29L);
            check(model.getSensitiveCount() == 73, family + " sensitive count");
            check(model.getResistantCount() == 17, family + " resistant count");
            check(model.getTotalCount() == 90, family + " total count");
            for (int value : model.getGrid()) {
                check(value >= 0 && value <= 2, family + " categorical grid");
            }
        }
    }

    private static void testMatchedOccupiedMask() {
        boolean[] reference = null;
        for (String family : FAMILIES) {
            ControlledModel model = model(20, 20, 0.1, 0.08, 0.01, 0.01, 0.75);
            model.reset(family, 73, 17, 11L, 29L, true);
            int[] grid = model.getGrid();
            boolean[] occupied = new boolean[grid.length];
            for (int i = 0; i < grid.length; i++) {
                occupied[i] = grid[i] != ControlledModel.EMPTY;
            }
            if (reference == null) {
                reference = occupied;
            } else {
                check(Arrays.equals(reference, occupied),
                        family + " reuses matched occupied-site mask");
            }
        }
    }

    private static void testDeterminism() {
        ControlledModel left = model(12, 9, 0.3, 0.2, 0.02, 0.03, 0.75);
        ControlledModel right = model(12, 9, 0.3, 0.2, 0.02, 0.03, 0.75);
        left.reset("resistant_dispersed", 50, 8, 123L, 456L);
        right.reset("resistant_dispersed", 50, 8, 123L, 456L);
        check(Arrays.equals(left.getGrid(), right.getGrid()), "deterministic reset");
        for (int i = 0; i < 25; i++) {
            double dose = (i % 3) / 2.0;
            left.step(dose);
            right.step(dose);
            check(Arrays.equals(left.getGrid(), right.getGrid()), "deterministic step " + i);
            check(left.csvRow(dose).equals(right.csvRow(dose)), "deterministic diagnostics " + i);
        }
    }

    private static void testNoMovement() {
        ControlledModel model = model(8, 8, 0.0, 0.0, 0.0, 0.0, 1.0);
        model.reset("random_mixed", 20, 10, 1L, 2L);
        int[] initial = model.getGrid();
        for (int i = 0; i < 20; i++) {
            model.step(i % 2);
        }
        check(Arrays.equals(initial, model.getGrid()), "zero-rate cells never move");
    }

    private static void testNoFluxAndBlockedDiagnostics() {
        ControlledModel model = model(1, 1, 1.0, 1.0, 0.0, 0.0, 0.0);
        model.reset("random_mixed", 0, 1, 1L, 1L);
        ControlledModel.Diagnostics d = model.step(0.0);
        check(model.getResistantCount() == 1, "boundary cell remains in grid");
        check(d.attemptedDivisionsResistant == 1, "resistant attempted division");
        check(d.blockedDivisionsResistant == 1, "boundary division blocked");
    }

    private static void testSensitiveOnlyDrugKill() {
        ControlledModel sensitive = model(2, 1, 1.0, 1.0, 0.0, 0.0, 1.0);
        sensitive.reset("resistant_core", 1, 0, 4L, 4L);
        ControlledModel.Diagnostics sd = sensitive.step(1.0);
        check(sensitive.getSensitiveCount() == 0, "drug kills sensitive cell");
        check(sd.drugDeathsSensitive == 1, "sensitive drug death recorded");
        check(sd.drugDeathsResistant == 0, "no resistant drug death recorded");

        ControlledModel resistant = model(2, 1, 1.0, 1.0, 0.0, 0.0, 1.0);
        resistant.reset("resistant_core", 0, 1, 4L, 4L);
        ControlledModel.Diagnostics rd = resistant.step(1.0);
        check(resistant.getResistantCount() == 2, "resistant cell unaffected by drug");
        check(rd.drugDeathsResistant == 0, "resistant drug death remains zero");
    }

    private static void testNaturalDeathDiagnostics() {
        ControlledModel model = model(2, 1, 0.0, 0.0, 1.0, 1.0, 1.0);
        model.reset("random_mixed", 1, 1, 9L, 9L);
        ControlledModel.Diagnostics d = model.step(1.0);
        check(model.getTotalCount() == 0, "natural death removes both phenotypes");
        check(d.naturalDeathsSensitive == 1, "sensitive natural death recorded");
        check(d.naturalDeathsResistant == 1, "resistant natural death recorded");
        check(d.drugDeathsSensitive == 0, "drug kill is division-conditional");
    }

    private static void testAdaptiveBoundary() {
        check(ControlledModel.paperAdaptiveDose(100, 100, 0.5, 0.0) == 0.0,
                "paper_adaptive uses strict N > N0");
        check(ControlledModel.paperAdaptiveDose(101, 100, 0.5, 0.0) == 1.0,
                "paper_adaptive starts above N0");
        check(ControlledModel.paperTextGeDose(100, 100, 0.5, 0.0) == 1.0,
                "paper_text_ge exposes inclusive variant");
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
