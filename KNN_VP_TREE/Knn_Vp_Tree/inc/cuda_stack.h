// C program for array implementation of stack 
#include <limits.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h>
#include "vptree.h"

// A structure to represent a stack 
typedef struct Stack { 
	int top; 
	unsigned capacity; 
	tree_array *array; 
}Stack; 

// function to create a stack of given capacity. It initializes size of 
// stack as 0 
__device__ Stack* createStack(unsigned capacity) 
{ 
	struct Stack* stack = (Stack*)malloc(sizeof(Stack)); 
	stack->capacity = capacity; 
	stack->top = -1; 
	stack->array = (tree_array*)malloc(stack->capacity * sizeof(tree_array));
	return stack; 
} 

// Stack is full when top is equal to the last index 
__device__ int isFull(struct Stack* stack) 
{ 
	return stack->top == stack->capacity - 1; 
} 

// Stack is empty when top is equal to -1 
__device__  int isEmpty(struct Stack* stack) 
{ 
	return stack->top == -1; 
} 

// Function to add an item to stack. It increases top by 1 
__device__  void push(struct Stack* stack, tree_array *node) 
{ 
	++stack->top;
	//printf("%d | %lf | %d\n",node->idx,node->median,node->array_idx );
	//printf("%d \n",stack->top );
	if (isFull(stack)) 
		return; 
	stack->array[stack->top].idx = node->idx;
	stack->array[stack->top].median = node->median;
	stack->array[stack->top].valid = node->valid;
	stack->array[stack->top].array_idx = node->array_idx;
} 

// Function to remove an item from stack. It decreases top by 1 
__device__  int pop(struct Stack* stack , tree_array *node) 
{ 
	if (isEmpty(stack)) 
		return 0;
	node->idx = stack->array[stack->top].idx;
	node->median = stack->array[stack->top].median;
	node->valid = stack->array[stack->top].valid;
	node->array_idx = stack->array[stack->top].array_idx;
	stack->top--;
	return 1; 
}


__device__ void deleteStack(struct Stack *stack)
{
	free(stack->array);
	free(stack);
}