#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <cfloat>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <glm/glm.hpp>

const double G = 0.0001;
const double RLIMIT = 0.03;
const double DOMAIN_SIZE = 4.0;

struct Body {
    int index;
    double x, y;
    double mass;
    double vx, vy;
};

struct Node {
    double mass;
    double com_x, com_y; // Center of Mass
    bool isLeaf;
    
    // Bounding box of this node (quadrant)
    double min_x, max_x, min_y, max_y;
    
    // Pointers to children: NW, NE, SW, SE
    Node *nw, *ne, *sw, *se;
    
    // If leaf, holds the body data (or pointer to it)
    const Body* body;

    Node(double minx, double maxx, double miny, double maxy) 
        : mass(0), com_x(0), com_y(0), isLeaf(true), 
          min_x(minx), max_x(maxx), min_y(miny), max_y(maxy),
          nw(nullptr), ne(nullptr), sw(nullptr), se(nullptr), body(nullptr) {}
};

MPI_Datatype mpi_body_type;

void create_mpi_type() {
    // We need to send the Body struct over MPI. 
    // Structure: int (1), double (5)
    int blocklengths[2] = {1, 5};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};

    displacements[0] = offsetof(Body, index);
    displacements[1] = offsetof(Body, x);

    MPI_Type_create_struct(2, blocklengths, displacements, types, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);
}

int getQuadrant(double x, double y, double mid_x, double mid_y) {
    if (x <= mid_x) {
        return (y <= mid_y) ? 2 : 0; // SW : NW
    } else {
        return (y <= mid_y) ? 3 : 1; // SE : NE
    }
}

void insert(Node* node, const Body* body) {
    // If body is lost (mass == -1), do not insert into tree
    if (body->mass == -1) return;

    // Update center of mass and total mass of the current node
    double total_mass = node->mass + body->mass;
    node->com_x = (node->com_x * node->mass + body->x * body->mass) / total_mass;
    node->com_y = (node->com_y * node->mass + body->y * body->mass) / total_mass;
    node->mass = total_mass;

    // If it's an internal node, recurse
    if (!node->isLeaf) {
        double mid_x = (node->min_x + node->max_x) / 2.0;
        double mid_y = (node->min_y + node->max_y) / 2.0;
        int quad = getQuadrant(body->x, body->y, mid_x, mid_y);

        if (quad == 0) insert(node->nw, body);
        else if (quad == 1) insert(node->ne, body);
        else if (quad == 2) insert(node->sw, body);
        else insert(node->se, body);
    } 
    // If it is a leaf
    else {
        // Case 1: Leaf is empty. Just put the body here.
        if (node->body == nullptr) {
            node->body = body;
        } 
        // Case 2: Leaf is occupied. We must split.
        else {
            const Body* existingBody = node->body;
            node->body = nullptr; // This node is no longer a leaf holding a body
            node->isLeaf = false;

            double mid_x = (node->min_x + node->max_x) / 2.0;
            double mid_y = (node->min_y + node->max_y) / 2.0;

            // Create children
            node->nw = new Node(node->min_x, mid_x, mid_y, node->max_y);
            node->ne = new Node(mid_x, node->max_x, mid_y, node->max_y);
            node->sw = new Node(node->min_x, mid_x, node->min_y, mid_y);
            node->se = new Node(mid_x, node->max_x, node->min_y, mid_y);

            // Re-insert the existing body into children
            insert(node, existingBody);
            // Insert the new body into children
            insert(node, body);
        }
    }
}

void freeTree(Node* node) {
    if (node == nullptr) return;
    freeTree(node->nw);
    freeTree(node->ne);
    freeTree(node->sw);
    freeTree(node->se);
    delete node;
}

void computeForce(Node* node, const Body* target, double theta, double& fx, double& fy) {
    if (node == nullptr || node->mass == 0) return;

    double dx = node->com_x - target->x;
    double dy = node->com_y - target->y;
    double distSq = dx*dx + dy*dy;
    double dist = sqrt(distSq);

    // If node contains the body itself, skip
    if (dist == 0) return;

    // Width of the region
    double s = node->max_x - node->min_x;

    // MAC: If node is far enough or is a leaf, compute force
    // Note: If it's a leaf, s/dist check doesn't matter, we MUST compute direct force
    if (node->isLeaf || (s / dist < theta)) {
        // Compute Force
        // Apply rlimit to avoid infinity
        double effective_dist = (dist < RLIMIT) ? RLIMIT : dist;
        double effective_dist_cubed = effective_dist * effective_dist * effective_dist;

        double f = (G * target->mass * node->mass) / effective_dist_cubed;
        
        fx += f * dx;
        fy += f * dy;
    } 
    else {
        // Recurse
        computeForce(node->nw, target, theta, fx, fy);
        computeForce(node->ne, target, theta, fx, fy);
        computeForce(node->sw, target, theta, fx, fy);
        computeForce(node->se, target, theta, fx, fy);
    }
}

void updateBody(Body& b, double fx, double fy, double dt) {
    if (b.mass == -1) return; // Ignore lost bodies

    double ax = fx / b.mass;
    double ay = fy / b.mass;

    // Leapfrog-Verlet Integration
    // P' = P + V*dt + 0.5*a*dt^2
    double next_x = b.x + b.vx * dt + 0.5 * ax * dt * dt;
    double next_y = b.y + b.vy * dt + 0.5 * ay * dt * dt;
    
    // V' = V + a*dt
    double next_vx = b.vx + ax * dt;
    double next_vy = b.vy + ay * dt;

    b.x = next_x;
    b.y = next_y;
    b.vx = next_vx;
    b.vy = next_vy;

    // Boundary check
    if (b.x < 0 || b.x > DOMAIN_SIZE || b.y < 0 || b.y > DOMAIN_SIZE) {
        b.mass = -1; // Mark as lost
    }
}

void readInput(const char* filename, std::vector<Body>& bodies) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening input file" << std::endl;
        exit(1);
    }
    int n;
    infile >> n;
    bodies.resize(n);
    for (int i = 0; i < n; i++) {
        infile >> bodies[i].index >> bodies[i].x >> bodies[i].y 
               >> bodies[i].mass >> bodies[i].vx >> bodies[i].vy;
    }
    infile.close();
}

void writeOutput(const char* filename, const std::vector<Body>& bodies) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening output file" << std::endl;
        exit(1);
    }
    outfile << bodies.size() << "\n";
    for (const auto& b : bodies) {
        outfile << b.index << " " << b.x << " " << b.y << " " 
                << b.mass << " " << b.vx << " " << b.vy << "\n";
    }
    outfile.close();
}

GLFWwindow* window;

// Coordinate Map: Domain(0, 4) -> OpenGL(-1, 1)
double toGlX(double x) { return 2.0 * x / DOMAIN_SIZE - 1.0; }
double toGlY(double y) { return 2.0 * y / DOMAIN_SIZE - 1.0; }

void drawParticle2D(double x_window, double y_window, double radius, float r, float g, float b) {
    int k = 0;
    float angle = 0.0f;
    glBegin(GL_TRIANGLE_FAN);
    glColor3f(r, g, b);
    glVertex2f(x_window, y_window);
    for(k=0; k<=20; k++){
        angle=(float)(k)/20*2*3.141592;
        glVertex2f(x_window + radius*cos(angle), y_window + radius*sin(angle));
    }
    glEnd();
}

void drawTree2D(Node* node) {
    if (!node || node->isLeaf) return;

    // Calculate split lines in GL coordinates
    double mid_x = (node->min_x + node->max_x) / 2.0;
    double mid_y = (node->min_y + node->max_y) / 2.0;

    double gl_min_x = toGlX(node->min_x);
    double gl_max_x = toGlX(node->max_x);
    double gl_min_y = toGlY(node->min_y);
    double gl_max_y = toGlY(node->max_y);
    double gl_mid_x = toGlX(mid_x);
    double gl_mid_y = toGlY(mid_y);

    glBegin(GL_LINES);
    glColor3f(0.2f, 0.2f, 0.2f); // Dark Grey for tree lines
    
    // Vertical divider
    glVertex2f(gl_mid_x, gl_min_y);
    glVertex2f(gl_mid_x, gl_max_y);
    
    // Horizontal divider
    glVertex2f(gl_min_x, gl_mid_y);
    glVertex2f(gl_max_x, gl_mid_y);
    glEnd();

    drawTree2D(node->nw);
    drawTree2D(node->ne);
    drawTree2D(node->sw);
    drawTree2D(node->se);
}

int initVisualization() {
    if( !glfwInit() ) {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }
    window = glfwCreateWindow( 600, 600, "N-Body Barnes Hut", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    return 0;
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    create_mpi_type();

    char* input_file = NULL;
    char* output_file = NULL;
    int steps = 0;
    double theta = 0.0;
    double dt = 0.0;
    bool visualization = false;

    int opt;
    while ((opt = getopt(argc, argv, "i:o:s:t:d:V")) != -1) {
        switch (opt) {
            case 'i': input_file = optarg; break;
            case 'o': output_file = optarg; break;
            case 's': steps = atoi(optarg); break;
            case 't': theta = atof(optarg); break;
            case 'd': dt = atof(optarg); break;
            case 'V': visualization = true; break;
            default: break;
        }
    }

    // Initialize Visualization (Only Rank 0)
    if (visualization && rank == 0) {
        if (initVisualization() != 0) {
            MPI_Finalize();
            return 1;
        }
    }

    std::vector<Body> bodies;
    if (rank == 0) {
        std::ifstream infile(input_file);
        int n;
        infile >> n;
        bodies.resize(n);
        for (int i = 0; i < n; i++) {
            infile >> bodies[i].index >> bodies[i].x >> bodies[i].y 
                   >> bodies[i].mass >> bodies[i].vx >> bodies[i].vy;
        }
    }

    int n = 0;
    if (rank == 0) n = bodies.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) bodies.resize(n);
    MPI_Bcast(bodies.data(), n, mpi_body_type, 0, MPI_COMM_WORLD);

    int count = n / size;
    int remainder = n % size;
    int start_idx = rank * count + (rank < remainder ? rank : remainder);
    int end_idx = start_idx + count + (rank < remainder ? 1 : 0);

    std::vector<int> recv_counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        int c = n / size;
        int rem = n % size;
        int s = i * c + (i < rem ? i : rem);
        int e = s + c + (i < rem ? 1 : 0);
        recv_counts[i] = e - s;
        displs[i] = s;
    }

    double start_time = MPI_Wtime();

    for (int step = 0; step < steps; step++) {
        
        // 1. Build Tree
        Node* root = new Node(0, DOMAIN_SIZE, 0, DOMAIN_SIZE);
        for (const auto& b : bodies) insert(root, &b);

        // 2. Visualization (Rank 0 only, at start of step)
        if (visualization && rank == 0) {
            glClear(GL_COLOR_BUFFER_BIT);

            // Draw Tree
            drawTree2D(root);

            // Draw Bodies
            for (const auto& b : bodies) {
                if (b.mass != -1) {
                    // Bright Cyan for bodies
                    drawParticle2D(toGlX(b.x), toGlY(b.y), 0.015, 0.0f, 1.0f, 1.0f);
                }
            }

            glfwSwapBuffers(window);
            glfwPollEvents();

            // Optional: Close if ESC pressed or window closed
            if (glfwWindowShouldClose(window)) {
                 // In a real app we might want to broadcast a stop signal
            }
        }

        // 3. Compute Forces & Update
        for (int i = start_idx; i < end_idx; i++) {
            if (bodies[i].mass == -1) continue;
            double fx = 0.0, fy = 0.0;
            computeForce(root, &bodies[i], theta, fx, fy);
            updateBody(bodies[i], fx, fy, dt);
        }

        // 4. Cleanup & Sync
        freeTree(root);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                       bodies.data(), recv_counts.data(), displs.data(), mpi_body_type, 
                       MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("%lf\n", end_time - start_time);
        std::ofstream outfile(output_file);
        outfile << bodies.size() << "\n";
        for (const auto& b : bodies) {
            outfile << b.index << " " << b.x << " " << b.y << " " 
                    << b.mass << " " << b.vx << " " << b.vy << "\n";
        }
        if (visualization) glfwTerminate();
    }

    MPI_Type_free(&mpi_body_type);
    MPI_Finalize();
    return 0;
}