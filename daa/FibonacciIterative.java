import java.util.Scanner;

public class FibonacciIterative {

    public static int fibonacci(int n) {
        if (n <= 1)
            return n;

        int a = 0, b = 1, c = 0;
        System.out.print(a + " " + b + " ");
        for (int i = 2; i <= n; i++) {
            c = a + b;
            System.out.print(c + " ");
            a = b;
            b = c;
        }

        return c;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter a number: ");
        int n = sc.nextInt();

        System.out.println("Fibonacci(" + n + ") = " + fibonacci(n));
    }
}
