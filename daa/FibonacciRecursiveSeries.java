import java.util.Scanner;

public class FibonacciRecursiveSeries {

    // Naive recursive function
    public static int fibonacci(int n) {
        if (n <= 1)
            return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    // Function to print full series
    public static void printSeries(int n) {
        System.out.print("Fibonacci Series: ");
        for (int i = 0; i <= n; i++) {
            System.out.print(fibonacci(i) + " ");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter n: ");
        int n = sc.nextInt();

        printSeries(n);

        System.out.println("Fibonacci(" + n + ") = " + fibonacci(n));
    }
}
