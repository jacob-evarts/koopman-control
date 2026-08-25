package strobl.control;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** Command-line and persistent line-protocol entry point. */
public final class ControlledCli {
    private static final String UPSTREAM_COMMIT =
            "aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2";

    private ControlledCli() {
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        Map<String, String> options = parseOptions(args);
        ControlledModel model = newModel(options);
        String mode = option(options, "mode", "serve");
        if ("serve".equals(mode)) {
            serve(model);
        } else if ("batch".equals(mode)) {
            batch(model, options);
        } else {
            throw new IllegalArgumentException("--mode must be serve or batch");
        }
    }

    private static ControlledModel newModel(Map<String, String> options) {
        return new ControlledModel(
                intOption(options, "width", 100),
                intOption(options, "height", 100),
                doubleOption(options, "dt", 1.0),
                doubleOption(options, "division-sensitive", 0.027),
                doubleOption(options, "division-resistant", 0.027),
                doubleOption(options, "death-sensitive", 0.0),
                doubleOption(options, "death-resistant", 0.0),
                doubleOption(options, "drug-kill", 0.75));
    }

    private static void batch(
            ControlledModel model, Map<String, String> options) throws Exception {
        String family = option(options, "family", "random_mixed");
        int sensitive = intOption(options, "sensitive", 180);
        int resistant = intOption(options, "resistant", 20);
        long simulationSeed = longOption(options, "simulation-seed", 7L);
        long icSeed = longOption(options, "ic-seed", simulationSeed);
        int steps = intOption(options, "steps", 10);
        if (steps < 0) {
            throw new IllegalArgumentException("--steps must be non-negative");
        }
        String policy = option(options, "policy", "external");
        double dose = doubleOption(options, "dose", 1.0);
        double withdrawalFraction = doubleOption(options, "withdrawal-fraction", 0.5);
        if (withdrawalFraction < 0.0 || withdrawalFraction > 1.0) {
            throw new IllegalArgumentException("--withdrawal-fraction must be in [0,1]");
        }
        model.reset(family, sensitive, resistant, simulationSeed, icSeed);

        String outputPath = option(options, "out", "-");
        PrintWriter output = "-".equals(outputPath)
                ? new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true)
                : new PrintWriter(new OutputStreamWriter(
                        new FileOutputStream(outputPath), StandardCharsets.UTF_8));
        try {
            output.println(ControlledModel.csvHeader());
            output.println(model.csvRow(dose));
            for (int i = 0; i < steps; i++) {
                if ("paper_adaptive".equals(policy)) {
                    dose = ControlledModel.paperAdaptiveDose(
                            model.getTotalCount(), model.getInitialTotal(),
                            withdrawalFraction, dose);
                } else if ("paper_text_ge".equals(policy)) {
                    dose = ControlledModel.paperTextGeDose(
                            model.getTotalCount(), model.getInitialTotal(),
                            withdrawalFraction, dose);
                } else if (!"external".equals(policy)) {
                    throw new IllegalArgumentException(
                            "--policy must be external, paper_adaptive, or paper_text_ge");
                }
                model.step(dose);
                output.println(model.csvRow(dose));
            }
        } finally {
            if (!"-".equals(outputPath)) {
                output.close();
            } else {
                output.flush();
            }
        }

        String gridOutputPath = options.get("grid-out");
        if (gridOutputPath != null) {
            PrintWriter gridOutput = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(gridOutputPath), StandardCharsets.UTF_8));
            try {
                writeGridCsv(gridOutput, model);
            } finally {
                gridOutput.close();
            }
        }
    }

    /**
     * Protocol commands (tab or whitespace separated):
     * RESET|INIT family sensitive resistant simulationSeed icSeed [sharedOccupiedMask]
     * STEP dose
     * COUNTS
     * GRID
     * QUIT
     */
    private static void serve(ControlledModel model) throws Exception {
        BufferedReader input = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintWriter output = new PrintWriter(
                new OutputStreamWriter(System.out, StandardCharsets.UTF_8), true);
        output.println("READY\tstrobl-controlled-v1\t" + UPSTREAM_COMMIT);
        String line;
        while ((line = input.readLine()) != null) {
            String trimmed = line.trim();
            if (trimmed.length() == 0 || trimmed.startsWith("#")) {
                continue;
            }
            String[] fields = trimmed.split("\\s+");
            String command = fields[0].toUpperCase(Locale.ROOT);
            try {
                if ("RESET".equals(command) || "INIT".equals(command)) {
                    if (fields.length != 6 && fields.length != 7) {
                        throw new IllegalArgumentException(
                                "expected 5 or 6 command arguments");
                    }
                    model.reset(
                            fields[1],
                            Integer.parseInt(fields[2]),
                            Integer.parseInt(fields[3]),
                            Long.parseLong(fields[4]),
                            Long.parseLong(fields[5]),
                            fields.length == 7 && Boolean.parseBoolean(fields[6]));
                    output.println(stateLine(model, Double.NaN));
                } else if ("STEP".equals(command)) {
                    requireFields(fields, 2);
                    double dose = Double.parseDouble(fields[1]);
                    model.step(dose);
                    output.println(stateLine(model, dose));
                } else if ("COUNTS".equals(command)) {
                    output.println(stateLine(model, Double.NaN));
                } else if ("GRID".equals(command)) {
                    output.println(gridLine(model));
                } else if ("QUIT".equals(command)) {
                    output.println("BYE");
                    return;
                } else {
                    output.println("ERROR\tunknown command");
                }
            } catch (RuntimeException error) {
                output.println("ERROR\t" + sanitize(error.getMessage()));
            }
        }
    }

    private static String stateLine(ControlledModel model, double dose) {
        ControlledModel.Diagnostics d = model.getDiagnostics();
        return String.format(
                Locale.ROOT,
                "STATE\tstep=%d\ttime=%.12g\tdose=%s\tsensitive=%d\tresistant=%d"
                        + "\ttotal=%d\tattempted_sensitive=%d\tattempted_resistant=%d"
                        + "\tblocked_sensitive=%d\tblocked_resistant=%d"
                        + "\tnatural_deaths_sensitive=%d\tnatural_deaths_resistant=%d"
                        + "\tdrug_deaths_sensitive=%d\tdrug_deaths_resistant=%d",
                Long.valueOf(model.getStepIndex()),
                Double.valueOf(model.getTime()),
                Double.isNaN(dose) ? "NA" : String.format(Locale.ROOT, "%.12g", dose),
                Integer.valueOf(model.getSensitiveCount()),
                Integer.valueOf(model.getResistantCount()),
                Integer.valueOf(model.getTotalCount()),
                Integer.valueOf(d.attemptedDivisionsSensitive),
                Integer.valueOf(d.attemptedDivisionsResistant),
                Integer.valueOf(d.blockedDivisionsSensitive),
                Integer.valueOf(d.blockedDivisionsResistant),
                Integer.valueOf(d.naturalDeathsSensitive),
                Integer.valueOf(d.naturalDeathsResistant),
                Integer.valueOf(d.drugDeathsSensitive),
                Integer.valueOf(d.drugDeathsResistant));
    }

    private static String gridLine(ControlledModel model) {
        StringBuilder result = new StringBuilder();
        result.append("GRID\twidth=").append(model.getWidth())
                .append("\theight=").append(model.getHeight()).append("\tvalues=");
        int[] grid = model.getGrid();
        for (int i = 0; i < grid.length; i++) {
            if (i > 0) {
                result.append(',');
            }
            result.append(grid[i]);
        }
        return result.toString();
    }

    private static void writeGridCsv(PrintWriter output, ControlledModel model) {
        output.println("x,y,state");
        int[] grid = model.getGrid();
        for (int i = 0; i < grid.length; i++) {
            output.println((i % model.getWidth()) + "," + (i / model.getWidth())
                    + "," + grid[i]);
        }
    }

    private static void requireFields(String[] fields, int expected) {
        if (fields.length != expected) {
            throw new IllegalArgumentException(
                    "expected " + (expected - 1) + " command arguments");
        }
    }

    private static String sanitize(String message) {
        if (message == null) {
            return "unspecified error";
        }
        return message.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> result = new HashMap<String, String>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException(
                        "arguments must be --name value pairs");
            }
            result.put(args[i].substring(2), args[++i]);
        }
        return result;
    }

    private static String option(
            Map<String, String> options, String name, String defaultValue) {
        String value = options.get(name);
        return value == null ? defaultValue : value;
    }

    private static int intOption(
            Map<String, String> options, String name, int defaultValue) {
        return Integer.parseInt(option(options, name, Integer.toString(defaultValue)));
    }

    private static long longOption(
            Map<String, String> options, String name, long defaultValue) {
        return Long.parseLong(option(options, name, Long.toString(defaultValue)));
    }

    private static double doubleOption(
            Map<String, String> options, String name, double defaultValue) {
        return Double.parseDouble(option(options, name, Double.toString(defaultValue)));
    }
}
