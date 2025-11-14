import java.util.*;

public class Knapsack01DP {

    public static int knapsack(int weights[], int values[], int capacity, int n) {
        int dp[][] = new int[n + 1][capacity + 1];

        // Build DP table
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {

                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                            values[i - 1] + dp[i - 1][w - weights[i - 1]],
                            dp[i - 1][w]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }

        return dp[n][capacity];
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        int[] weights = new int[n];
        int[] values = new int[n];

        System.out.println("Enter weight and value of each item:");
        for (int i = 0; i < n; i++) {
            System.out.print("Weight of item " + (i + 1) + ": ");
            weights[i] = sc.nextInt();

            System.out.print("Value of item " + (i + 1) + ": ");
            values[i] = sc.nextInt();
        }

        System.out.print("Enter Knapsack Capacity: ");
        int capacity = sc.nextInt();

        int maxProfit = knapsack(weights, values, capacity, n);

        System.out.println("\nMaximum value in Knapsack = " + maxProfit);
    }
}
