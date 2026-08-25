package strobl.control;

import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Headless, externally controlled adaptation of the Strobl et al. on-lattice
 * model. Grid values are exactly 0 (empty), 1 (sensitive), or 2 (resistant).
 */
public final class ControlledModel {
    public static final int EMPTY = 0;
    public static final int SENSITIVE = 1;
    public static final int RESISTANT = 2;

    private final int width;
    private final int height;
    private final double dt;
    private final double divisionSensitive;
    private final double divisionResistant;
    private final double deathSensitive;
    private final double deathResistant;
    private final double drugKillProportion;
    private final byte[] grid;
    private final ArrayList<Integer> agents = new ArrayList<Integer>();

    private Random simulationRandom;
    private long stepIndex;
    private int initialTotal;
    private int sensitiveCount;
    private int resistantCount;
    private Diagnostics diagnostics = new Diagnostics();

    public ControlledModel(
            int width,
            int height,
            double dt,
            double divisionSensitive,
            double divisionResistant,
            double deathSensitive,
            double deathResistant,
            double drugKillProportion) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("grid dimensions must be positive");
        }
        requireProbabilityRate("dt", dt, false);
        requireProbabilityRate("divisionSensitive", divisionSensitive, true);
        requireProbabilityRate("divisionResistant", divisionResistant, true);
        requireProbabilityRate("deathSensitive", deathSensitive, true);
        requireProbabilityRate("deathResistant", deathResistant, true);
        if (!Double.isFinite(drugKillProportion)
                || drugKillProportion < 0.0
                || drugKillProportion > 1.0) {
            throw new IllegalArgumentException("drugKillProportion must be in [0,1]");
        }
        if ((divisionSensitive + deathSensitive) * dt > 1.0
                || (divisionResistant + deathResistant) * dt > 1.0) {
            throw new IllegalArgumentException("(division + death) * dt must be <= 1");
        }
        this.width = width;
        this.height = height;
        this.dt = dt;
        this.divisionSensitive = divisionSensitive;
        this.divisionResistant = divisionResistant;
        this.deathSensitive = deathSensitive;
        this.deathResistant = deathResistant;
        this.drugKillProportion = drugKillProportion;
        this.grid = new byte[width * height];
    }

    private static void requireProbabilityRate(String name, double value, boolean allowZero) {
        if (!Double.isFinite(value) || value < 0.0 || (!allowZero && value == 0.0)) {
            throw new IllegalArgumentException(name + " must be finite and "
                    + (allowZero ? "non-negative" : "positive"));
        }
    }

    /**
     * Reset both random streams and initialize one of the supported IC families.
     */
    public void reset(
            String family,
            int requestedSensitive,
            int requestedResistant,
            long simulationSeed,
            long initialConditionSeed) {
        reset(
                family,
                requestedSensitive,
                requestedResistant,
                simulationSeed,
                initialConditionSeed,
                false);
    }

    /**
     * Reset with an optional common compact occupied-site mask. This is used by
     * matched-state episodes so only resistant-cell arrangement differs.
     */
    public void reset(
            String family,
            int requestedSensitive,
            int requestedResistant,
            long simulationSeed,
            long initialConditionSeed,
            boolean sharedOccupiedMask) {
        if (requestedSensitive < 0 || requestedResistant < 0
                || requestedSensitive + requestedResistant > grid.length) {
            throw new IllegalArgumentException("requested counts do not fit the grid");
        }
        for (int i = 0; i < grid.length; i++) {
            grid[i] = EMPTY;
        }
        agents.clear();
        simulationRandom = new Random(simulationSeed);
        Random icRandom = new Random(initialConditionSeed);
        stepIndex = 0L;
        diagnostics = new Diagnostics();

        int total = requestedSensitive + requestedResistant;
        ArrayList<Integer> sites = allSites();
        Collections.shuffle(sites, icRandom);
        if ("random_mixed".equals(family) && !sharedOccupiedMask) {
            initializeRandomMixed(sites, total, requestedSensitive, icRandom);
        } else {
            final double centerX = (width - 1) / 2.0;
            final double centerY = (height - 1) / 2.0;
            Collections.sort(sites, new Comparator<Integer>() {
                public int compare(Integer left, Integer right) {
                    return Double.compare(
                            squaredDistance(left.intValue(), centerX, centerY),
                            squaredDistance(right.intValue(), centerX, centerY));
                }
            });
            ArrayList<Integer> occupied =
                    new ArrayList<Integer>(sites.subList(0, total));
            for (Integer site : occupied) {
                grid[site.intValue()] = SENSITIVE;
            }
            if ("random_mixed".equals(family)) {
                ArrayList<Integer> shuffledOccupied = new ArrayList<Integer>(occupied);
                Collections.shuffle(shuffledOccupied, icRandom);
                for (int i = 0; i < requestedResistant; i++) {
                    grid[shuffledOccupied.get(i).intValue()] = RESISTANT;
                }
            } else if ("resistant_core".equals(family)) {
                placeCore(occupied, requestedResistant);
            } else if ("resistant_edge".equals(family)) {
                placeEdge(occupied, requestedResistant);
            } else if ("resistant_dispersed".equals(family)) {
                placeDispersed(occupied, requestedResistant, icRandom);
            } else if ("two_resistant_nests".equals(family)) {
                placeTwoNests(occupied, requestedResistant);
            } else {
                throw new IllegalArgumentException("unknown IC family: " + family);
            }
        }
        rebuildAgentsAndCounts();
        initialTotal = total;
        if (sensitiveCount != requestedSensitive || resistantCount != requestedResistant) {
            throw new IllegalStateException("initializer failed to preserve exact counts");
        }
    }

    private ArrayList<Integer> allSites() {
        ArrayList<Integer> sites = new ArrayList<Integer>(grid.length);
        for (int i = 0; i < grid.length; i++) {
            sites.add(Integer.valueOf(i));
        }
        return sites;
    }

    private void initializeRandomMixed(
            List<Integer> shuffledSites, int total, int nSensitive, Random random) {
        ArrayList<Byte> phenotypes = new ArrayList<Byte>(total);
        for (int i = 0; i < nSensitive; i++) {
            phenotypes.add(Byte.valueOf((byte) SENSITIVE));
        }
        while (phenotypes.size() < total) {
            phenotypes.add(Byte.valueOf((byte) RESISTANT));
        }
        Collections.shuffle(phenotypes, random);
        for (int i = 0; i < total; i++) {
            grid[shuffledSites.get(i).intValue()] = phenotypes.get(i).byteValue();
        }
    }

    private void placeCore(List<Integer> occupied, int nResistant) {
        List<Integer> connected = connectedOrder(occupied.get(0).intValue());
        for (int i = 0; i < nResistant; i++) {
            grid[connected.get(i).intValue()] = RESISTANT;
        }
    }

    private void placeEdge(List<Integer> occupied, int nResistant) {
        int seed = occupied.get(0).intValue();
        for (Integer candidate : occupied) {
            int site = candidate.intValue();
            if (site % width > seed % width
                    || (site % width == seed % width
                    && Math.abs(site / width - height / 2)
                    < Math.abs(seed / width - height / 2))) {
                seed = site;
            }
        }
        List<Integer> connected = connectedOrder(seed);
        for (int i = 0; i < nResistant; i++) {
            grid[connected.get(i).intValue()] = RESISTANT;
        }
    }

    private void placeDispersed(
            List<Integer> occupied, int nResistant, Random random) {
        if (nResistant == 0) {
            return;
        }
        ArrayList<Integer> candidates = new ArrayList<Integer>(occupied);
        Collections.shuffle(candidates, random);
        ArrayList<Integer> selected = new ArrayList<Integer>();
        selected.add(candidates.remove(0));
        while (selected.size() < nResistant) {
            int bestIndex = 0;
            int bestDistance = -1;
            for (int i = 0; i < candidates.size(); i++) {
                int minimumDistance = Integer.MAX_VALUE;
                for (Integer chosen : selected) {
                    minimumDistance = Math.min(
                            minimumDistance,
                            manhattan(candidates.get(i).intValue(), chosen.intValue()));
                }
                if (minimumDistance > bestDistance) {
                    bestDistance = minimumDistance;
                    bestIndex = i;
                }
            }
            selected.add(candidates.remove(bestIndex));
        }
        for (Integer site : selected) {
            grid[site.intValue()] = RESISTANT;
        }
    }

    private void placeTwoNests(List<Integer> occupied, int nResistant) {
        final int leftX = width / 3;
        final int rightX = (2 * width) / 3;
        final int centerY = height / 2;
        int leftSeed = nearestOccupied(occupied, leftX, centerY);
        int rightSeed = nearestOccupied(occupied, rightX, centerY);
        List<Integer> left = connectedOrder(leftSeed);
        List<Integer> right = connectedOrder(rightSeed);
        boolean[] selected = new boolean[grid.length];
        int leftIndex = 0;
        int rightIndex = 0;
        int placed = 0;
        while (placed < nResistant) {
            List<Integer> order = placed % 2 == 0 ? left : right;
            int index = placed % 2 == 0 ? leftIndex : rightIndex;
            while (index < order.size() && selected[order.get(index).intValue()]) {
                index++;
            }
            int site = order.get(index).intValue();
            if (placed % 2 == 0) {
                leftIndex = index + 1;
            } else {
                rightIndex = index + 1;
            }
            selected[site] = true;
            grid[site] = RESISTANT;
            placed++;
        }
    }

    private int nearestOccupied(List<Integer> occupied, int targetX, int targetY) {
        int best = occupied.get(0).intValue();
        int bestDistance = Integer.MAX_VALUE;
        for (Integer candidate : occupied) {
            int site = candidate.intValue();
            int distance = Math.abs(site % width - targetX)
                    + Math.abs(site / width - targetY);
            if (distance < bestDistance) {
                best = site;
                bestDistance = distance;
            }
        }
        return best;
    }

    private List<Integer> connectedOrder(int seed) {
        ArrayList<Integer> order = new ArrayList<Integer>();
        ArrayDeque<Integer> queue = new ArrayDeque<Integer>();
        boolean[] seen = new boolean[grid.length];
        queue.add(Integer.valueOf(seed));
        seen[seed] = true;
        while (!queue.isEmpty()) {
            int site = queue.removeFirst().intValue();
            order.add(Integer.valueOf(site));
            int x = site % width;
            int y = site / width;
            int[] neighbors = new int[] {
                x > 0 ? site - 1 : -1,
                x + 1 < width ? site + 1 : -1,
                y > 0 ? site - width : -1,
                y + 1 < height ? site + width : -1
            };
            for (int neighbor : neighbors) {
                if (neighbor >= 0 && !seen[neighbor] && grid[neighbor] != EMPTY) {
                    seen[neighbor] = true;
                    queue.addLast(Integer.valueOf(neighbor));
                }
            }
        }
        if (order.size() != agentsExpectedDuringInitialization()) {
            throw new IllegalStateException("occupied-site mask is not connected");
        }
        return order;
    }

    private int agentsExpectedDuringInitialization() {
        int count = 0;
        for (byte value : grid) {
            if (value != EMPTY) {
                count++;
            }
        }
        return count;
    }

    private void rebuildAgentsAndCounts() {
        agents.clear();
        sensitiveCount = 0;
        resistantCount = 0;
        for (int i = 0; i < grid.length; i++) {
            if (grid[i] == SENSITIVE) {
                sensitiveCount++;
                agents.add(Integer.valueOf(i));
            } else if (grid[i] == RESISTANT) {
                resistantCount++;
                agents.add(Integer.valueOf(i));
            }
        }
    }

    /**
     * Advance exactly one externally controlled step with a homogeneous dose.
     * No movement or internally selected treatment/progression behavior occurs.
     */
    public Diagnostics step(double dose) {
        if (simulationRandom == null) {
            throw new IllegalStateException("reset must be called before step");
        }
        if (!Double.isFinite(dose) || dose < 0.0 || dose > 1.0) {
            throw new IllegalArgumentException("dose must be finite and in [0,1]");
        }
        Diagnostics next = new Diagnostics();
        ArrayList<Integer> currentAgents = new ArrayList<Integer>(agents);
        for (Integer boxedSite : currentAgents) {
            int site = boxedSite.intValue();
            int phenotype = grid[site];
            if (phenotype == EMPTY) {
                continue;
            }
            double division = phenotype == SENSITIVE
                    ? divisionSensitive : divisionResistant;
            double naturalDeath = phenotype == SENSITIVE
                    ? deathSensitive : deathResistant;
            double totalPropensity = (division + naturalDeath) * dt;
            if (totalPropensity == 0.0 || simulationRandom.nextDouble() >= totalPropensity) {
                continue;
            }
            if (simulationRandom.nextDouble() < division / (division + naturalDeath)) {
                next.incrementAttempted(phenotype);
                int[] emptyNeighbors = emptyNeighbors(site);
                if (emptyNeighbors.length == 0) {
                    next.incrementBlocked(phenotype);
                    continue;
                }
                if (phenotype == SENSITIVE
                        && simulationRandom.nextDouble() < drugKillProportion * dose) {
                    removeAgent(site, phenotype);
                    next.incrementDrugDeath(phenotype);
                } else {
                    int daughterSite =
                            emptyNeighbors[simulationRandom.nextInt(emptyNeighbors.length)];
                    addAgent(daughterSite, phenotype);
                }
            } else {
                removeAgent(site, phenotype);
                next.incrementNaturalDeath(phenotype);
            }
        }
        Collections.shuffle(agents, simulationRandom);
        stepIndex++;
        diagnostics = next;
        return next.copy();
    }

    private int[] emptyNeighbors(int site) {
        int x = site % width;
        int y = site / width;
        int[] scratch = new int[4];
        int count = 0;
        if (x > 0 && grid[site - 1] == EMPTY) {
            scratch[count++] = site - 1;
        }
        if (x + 1 < width && grid[site + 1] == EMPTY) {
            scratch[count++] = site + 1;
        }
        if (y > 0 && grid[site - width] == EMPTY) {
            scratch[count++] = site - width;
        }
        if (y + 1 < height && grid[site + width] == EMPTY) {
            scratch[count++] = site + width;
        }
        int[] result = new int[count];
        System.arraycopy(scratch, 0, result, 0, count);
        return result;
    }

    private void addAgent(int site, int phenotype) {
        grid[site] = (byte) phenotype;
        agents.add(Integer.valueOf(site));
        if (phenotype == SENSITIVE) {
            sensitiveCount++;
        } else {
            resistantCount++;
        }
    }

    private void removeAgent(int site, int phenotype) {
        grid[site] = EMPTY;
        agents.remove(Integer.valueOf(site));
        if (phenotype == SENSITIVE) {
            sensitiveCount--;
        } else {
            resistantCount--;
        }
    }

    private double squaredDistance(int site, double x, double y) {
        double dx = site % width - x;
        double dy = site / width - y;
        return dx * dx + dy * dy;
    }

    private int manhattan(int left, int right) {
        return Math.abs(left % width - right % width)
                + Math.abs(left / width - right / width);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public long getStepIndex() {
        return stepIndex;
    }

    public double getTime() {
        return stepIndex * dt;
    }

    public int getSensitiveCount() {
        return sensitiveCount;
    }

    public int getResistantCount() {
        return resistantCount;
    }

    public int getTotalCount() {
        return sensitiveCount + resistantCount;
    }

    public int getInitialTotal() {
        return initialTotal;
    }

    public int[] getGrid() {
        int[] result = new int[grid.length];
        for (int i = 0; i < grid.length; i++) {
            result[i] = grid[i];
        }
        return result;
    }

    public Diagnostics getDiagnostics() {
        return diagnostics.copy();
    }

    public static double paperAdaptiveDose(
            int population, int initialPopulation, double withdrawalFraction, double previousDose) {
        if (population > initialPopulation) {
            return 1.0; // Strict N > N0, matching the released source.
        }
        if (population < (1.0 - withdrawalFraction) * initialPopulation) {
            return 0.0;
        }
        return previousDose > 0.0 ? 1.0 : 0.0;
    }

    public static double paperTextGeDose(
            int population, int initialPopulation, double withdrawalFraction, double previousDose) {
        if (population >= initialPopulation) {
            return 1.0;
        }
        if (population < (1.0 - withdrawalFraction) * initialPopulation) {
            return 0.0;
        }
        return previousDose > 0.0 ? 1.0 : 0.0;
    }

    public static String csvHeader() {
        return "step,time,dose,sensitive,resistant,total,"
                + "attempted_divisions_sensitive,attempted_divisions_resistant,"
                + "blocked_divisions_sensitive,blocked_divisions_resistant,"
                + "natural_deaths_sensitive,natural_deaths_resistant,"
                + "drug_deaths_sensitive,drug_deaths_resistant";
    }

    public String csvRow(double dose) {
        return String.format(
                Locale.ROOT,
                "%d,%.12g,%.12g,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                Long.valueOf(stepIndex),
                Double.valueOf(getTime()),
                Double.valueOf(dose),
                Integer.valueOf(sensitiveCount),
                Integer.valueOf(resistantCount),
                Integer.valueOf(getTotalCount()),
                Integer.valueOf(diagnostics.attemptedDivisionsSensitive),
                Integer.valueOf(diagnostics.attemptedDivisionsResistant),
                Integer.valueOf(diagnostics.blockedDivisionsSensitive),
                Integer.valueOf(diagnostics.blockedDivisionsResistant),
                Integer.valueOf(diagnostics.naturalDeathsSensitive),
                Integer.valueOf(diagnostics.naturalDeathsResistant),
                Integer.valueOf(diagnostics.drugDeathsSensitive),
                Integer.valueOf(diagnostics.drugDeathsResistant));
    }

    public static final class Diagnostics {
        public int attemptedDivisionsSensitive;
        public int attemptedDivisionsResistant;
        public int blockedDivisionsSensitive;
        public int blockedDivisionsResistant;
        public int naturalDeathsSensitive;
        public int naturalDeathsResistant;
        public int drugDeathsSensitive;
        public int drugDeathsResistant;

        private void incrementAttempted(int phenotype) {
            if (phenotype == SENSITIVE) {
                attemptedDivisionsSensitive++;
            } else {
                attemptedDivisionsResistant++;
            }
        }

        private void incrementBlocked(int phenotype) {
            if (phenotype == SENSITIVE) {
                blockedDivisionsSensitive++;
            } else {
                blockedDivisionsResistant++;
            }
        }

        private void incrementNaturalDeath(int phenotype) {
            if (phenotype == SENSITIVE) {
                naturalDeathsSensitive++;
            } else {
                naturalDeathsResistant++;
            }
        }

        private void incrementDrugDeath(int phenotype) {
            if (phenotype == SENSITIVE) {
                drugDeathsSensitive++;
            } else {
                drugDeathsResistant++;
            }
        }

        private Diagnostics copy() {
            Diagnostics copy = new Diagnostics();
            copy.attemptedDivisionsSensitive = attemptedDivisionsSensitive;
            copy.attemptedDivisionsResistant = attemptedDivisionsResistant;
            copy.blockedDivisionsSensitive = blockedDivisionsSensitive;
            copy.blockedDivisionsResistant = blockedDivisionsResistant;
            copy.naturalDeathsSensitive = naturalDeathsSensitive;
            copy.naturalDeathsResistant = naturalDeathsResistant;
            copy.drugDeathsSensitive = drugDeathsSensitive;
            copy.drugDeathsResistant = drugDeathsResistant;
            return copy;
        }
    }
}
