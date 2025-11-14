import java.util.*;

class Item {
    int weight;
    int value;
    double ratio;

    Item(int weight, int value) {
        this.weight = weight;
        this.value = value;
        this.ratio = (double) value / weight;
    }
}

public class FractionalKnapsack {

    public static double solveKnapsack(List<Item> items, int capacity) {
        // Sort items by value/weight ratio in descending order
        items.sort((a, b) -> Double.compare(b.ratio, a.ratio));

        double totalValue = 0.0;
        int currentWeight = 0;

        for (Item item : items) {
            if (currentWeight + item.weight <= capacity) {
                // take whole item
                currentWeight += item.weight;
                totalValue += item.value;
            } else {
                // take fractional part
                int remaining = capacity - currentWeight;
                totalValue += item.ratio * remaining;
                break;
            }
        }

        return totalValue;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of items: ");
        int n = sc.nextInt();

        List<Item> items = new ArrayList<>();

        System.out.println("Enter weight and value of each item:");
        for (int i = 0; i < n; i++) {
            System.out.print("Item " + (i + 1) + " Weight: ");
            int w = sc.nextInt();

            System.out.print("Item " + (i + 1) + " Value: ");
            int v = sc.nextInt();

            items.add(new Item(w, v));
        }

        System.out.print("Enter Knapsack Capacity: ");
        int capacity = sc.nextInt();

        double maxValue = solveKnapsack(items, capacity);

        System.out.println("\nMaximum value in Knapsack = " + maxValue);
    }
}
