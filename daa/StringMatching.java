import java.util.*;

public class StringMatching {

    // ------------------ NAIVE STRING MATCHING ------------------
    public static void naiveSearch(String text, String pattern) {
        int n = text.length();
        int m = pattern.length();

        System.out.println("\nNaive String Matching:");
        boolean found = false;

        for (int i = 0; i <= n - m; i++) {
            int j;
            for (j = 0; j < m; j++) {
                if (text.charAt(i + j) != pattern.charAt(j))
                    break;
            }
            if (j == m) {
                System.out.println("Pattern found at index " + i);
                found = true;
            }
        }
        if (!found)
            System.out.println("Pattern Not Found");
    }

    // ------------------ RABIN-KARP ALGORITHM ------------------
    public static void rabinKarpSearch(String text, String pattern) {
        int d = 256; // number of characters in ASCII
        int q = 101; // a prime number
        int n = text.length();
        int m = pattern.length();

        int p = 0; // hash value for pattern
        int t = 0; // hash value for text window
        int h = 1;

        System.out.println("\nRabin-Karp Matching:");
        boolean found = false;

        // h = (d^(m-1)) % q
        for (int i = 0; i < m - 1; i++)
            h = (h * d) % q;

        // Calculate hash of pattern and first window
        for (int i = 0; i < m; i++) {
            p = (d * p + pattern.charAt(i)) % q;
            t = (d * t + text.charAt(i)) % q;
        }

        // Slide the window over text
        for (int i = 0; i <= n - m; i++) {

            // If hash matches, check characters
            if (p == t) {
                int j;
                for (j = 0; j < m; j++) {
                    if (text.charAt(i + j) != pattern.charAt(j))
                        break;
                }
                if (j == m) {
                    System.out.println("Pattern found at index " + i);
                    found = true;
                }
            }

            // Calculate next window hash
            if (i < n - m) {
                t = (d * (t - text.charAt(i) * h) + text.charAt(i + 1)) % q;

                if (t < 0)
                    t = (t + q);
            }
        }

        if (!found)
            System.out.println("Pattern Not Found");
    }

    // ------------------ MAIN ------------------
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        System.out.println("Enter text: ");
        String text = sc.nextLine();

        System.out.println("Enter pattern: ");
        String pattern = sc.nextLine();

        naiveSearch(text, pattern);
        rabinKarpSearch(text, pattern);
    }
}
