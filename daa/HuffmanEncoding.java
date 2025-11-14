import java.util.*;

class HuffmanNode {
    int frequency;
    char character;
    HuffmanNode left, right;

    HuffmanNode(char character, int frequency) {
        this.character = character;
        this.frequency = frequency;
    }
}

public class HuffmanEncoding {

    // Recursive function to generate Huffman Codes
    public static void generateCodes(HuffmanNode root, String code, Map<Character, String> huffmanCodeMap) {
        if (root == null)
            return;

        // Leaf node -> store the code
        if (root.left == null && root.right == null) {
            huffmanCodeMap.put(root.character, code);
            return;
        }

        generateCodes(root.left, code + "0", huffmanCodeMap);
        generateCodes(root.right, code + "1", huffmanCodeMap);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Step 1: Input the string
        System.out.print("Enter a string: ");
        String text = sc.nextLine();

        // Step 2: Count frequency of each character
        Map<Character, Integer> freq = new HashMap<>();
        for (char c : text.toCharArray()) {
            freq.put(c, freq.getOrDefault(c, 0) + 1);
        }

        // Step 3: PriorityQueue (Min-Heap) based on frequency
        PriorityQueue<HuffmanNode> pq = new PriorityQueue<>(Comparator.comparingInt(n -> n.frequency));

        // Step 4: Create leaf nodes and add to min heap
        for (Map.Entry<Character, Integer> entry : freq.entrySet()) {
            pq.add(new HuffmanNode(entry.getKey(), entry.getValue()));
        }

        // Step 5: Build Huffman Tree using Greedy Strategy
        while (pq.size() > 1) {
            HuffmanNode left = pq.poll(); // smallest freq
            HuffmanNode right = pq.poll(); // second smallest freq

            HuffmanNode parent = new HuffmanNode('-', left.frequency + right.frequency);
            parent.left = left;
            parent.right = right;

            pq.add(parent);
        }

        HuffmanNode root = pq.peek();

        // Step 6: Generate Huffman Codes
        Map<Character, String> huffmanCodeMap = new HashMap<>();
        generateCodes(root, "", huffmanCodeMap);

        System.out.println("\nHuffman Codes:");
        for (Map.Entry<Character, String> entry : huffmanCodeMap.entrySet()) {
            System.out.println(entry.getKey() + " : " + entry.getValue());
        }

        // Step 7: Encode the input string
        StringBuilder encodedText = new StringBuilder();
        for (char c : text.toCharArray()) {
            encodedText.append(huffmanCodeMap.get(c));
        }

        System.out.println("\nOriginal String: " + text);
        System.out.println("Encoded String: " + encodedText.toString());
    }
}
